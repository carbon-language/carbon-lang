//===-- ObjectFilePECOFF.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectFilePECOFF.h"
#include "PECallFrameInfo.h"
#include "WindowsMiniDump.h"

#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"
#include "llvm/BinaryFormat/COFF.h"

#include "llvm/Object/COFFImportFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#define IMAGE_DOS_SIGNATURE 0x5A4D    // MZ
#define IMAGE_NT_SIGNATURE 0x00004550 // PE00
#define OPT_HEADER_MAGIC_PE32 0x010b
#define OPT_HEADER_MAGIC_PE32_PLUS 0x020b

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectFilePECOFF)

static UUID GetCoffUUID(llvm::object::COFFObjectFile &coff_obj) {
  const llvm::codeview::DebugInfo *pdb_info = nullptr;
  llvm::StringRef pdb_file;

  if (!coff_obj.getDebugPDBInfo(pdb_info, pdb_file) && pdb_info) {
    if (pdb_info->PDB70.CVSignature == llvm::OMF::Signature::PDB70) {
      UUID::CvRecordPdb70 info;
      memcpy(&info.Uuid, pdb_info->PDB70.Signature, sizeof(info.Uuid));
      info.Age = pdb_info->PDB70.Age;
      return UUID::fromCvRecord(info);
    }
  }

  return UUID();
}

char ObjectFilePECOFF::ID;

void ObjectFilePECOFF::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      CreateMemoryInstance, GetModuleSpecifications, SaveCore);
}

void ObjectFilePECOFF::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef ObjectFilePECOFF::GetPluginDescriptionStatic() {
  return "Portable Executable and Common Object File Format object file reader "
         "(32 and 64 bit)";
}

ObjectFile *ObjectFilePECOFF::CreateInstance(const lldb::ModuleSP &module_sp,
                                             DataBufferSP &data_sp,
                                             lldb::offset_t data_offset,
                                             const lldb_private::FileSpec *file_p,
                                             lldb::offset_t file_offset,
                                             lldb::offset_t length) {
  FileSpec file = file_p ? *file_p : FileSpec();
  if (!data_sp) {
    data_sp = MapFileData(file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }

  if (!ObjectFilePECOFF::MagicBytesMatch(data_sp))
    return nullptr;

  // Update the data to contain the entire file if it doesn't already
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(file, length, file_offset);
    if (!data_sp)
      return nullptr;
  }

  auto objfile_up = std::make_unique<ObjectFilePECOFF>(
      module_sp, data_sp, data_offset, file_p, file_offset, length);
  if (!objfile_up || !objfile_up->ParseHeader())
    return nullptr;

  // Cache coff binary.
  if (!objfile_up->CreateBinary())
    return nullptr;
  return objfile_up.release();
}

ObjectFile *ObjectFilePECOFF::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
    const lldb::ProcessSP &process_sp, lldb::addr_t header_addr) {
  if (!data_sp || !ObjectFilePECOFF::MagicBytesMatch(data_sp))
    return nullptr;
  auto objfile_up = std::make_unique<ObjectFilePECOFF>(
      module_sp, data_sp, process_sp, header_addr);
  if (objfile_up.get() && objfile_up->ParseHeader()) {
    return objfile_up.release();
  }
  return nullptr;
}

size_t ObjectFilePECOFF::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t length, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();
  if (!data_sp || !ObjectFilePECOFF::MagicBytesMatch(data_sp))
    return initial_count;

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_OBJECT));

  if (data_sp->GetByteSize() < length)
    if (DataBufferSP full_sp = MapFileData(file, -1, file_offset))
      data_sp = std::move(full_sp);
  auto binary = llvm::object::createBinary(llvm::MemoryBufferRef(
      toStringRef(data_sp->GetData()), file.GetFilename().GetStringRef()));

  if (!binary) {
    LLDB_LOG_ERROR(log, binary.takeError(),
                   "Failed to create binary for file ({1}): {0}", file);
    return initial_count;
  }

  auto *COFFObj = llvm::dyn_cast<llvm::object::COFFObjectFile>(binary->get());
  if (!COFFObj)
    return initial_count;

  ModuleSpec module_spec(file);
  ArchSpec &spec = module_spec.GetArchitecture();
  lldb_private::UUID &uuid = module_spec.GetUUID();
  if (!uuid.IsValid())
    uuid = GetCoffUUID(*COFFObj);

  switch (COFFObj->getMachine()) {
  case MachineAmd64:
    spec.SetTriple("x86_64-pc-windows");
    specs.Append(module_spec);
    break;
  case MachineX86:
    spec.SetTriple("i386-pc-windows");
    specs.Append(module_spec);
    spec.SetTriple("i686-pc-windows");
    specs.Append(module_spec);
    break;
  case MachineArmNt:
    spec.SetTriple("armv7-pc-windows");
    specs.Append(module_spec);
    break;
  case MachineArm64:
    spec.SetTriple("aarch64-pc-windows");
    specs.Append(module_spec);
    break;
  default:
    break;
  }

  return specs.GetSize() - initial_count;
}

bool ObjectFilePECOFF::SaveCore(const lldb::ProcessSP &process_sp,
                                const lldb_private::FileSpec &outfile,
                                lldb::SaveCoreStyle &core_style,
                                lldb_private::Status &error) {
  core_style = eSaveCoreFull;
  return SaveMiniDump(process_sp, outfile, error);
}

bool ObjectFilePECOFF::MagicBytesMatch(DataBufferSP &data_sp) {
  DataExtractor data(data_sp, eByteOrderLittle, 4);
  lldb::offset_t offset = 0;
  uint16_t magic = data.GetU16(&offset);
  return magic == IMAGE_DOS_SIGNATURE;
}

lldb::SymbolType ObjectFilePECOFF::MapSymbolType(uint16_t coff_symbol_type) {
  // TODO:  We need to complete this mapping of COFF symbol types to LLDB ones.
  // For now, here's a hack to make sure our function have types.
  const auto complex_type =
      coff_symbol_type >> llvm::COFF::SCT_COMPLEX_TYPE_SHIFT;
  if (complex_type == llvm::COFF::IMAGE_SYM_DTYPE_FUNCTION) {
    return lldb::eSymbolTypeCode;
  }
  return lldb::eSymbolTypeInvalid;
}

bool ObjectFilePECOFF::CreateBinary() {
  if (m_binary)
    return true;

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_OBJECT));

  auto binary = llvm::object::createBinary(llvm::MemoryBufferRef(
      toStringRef(m_data.GetData()), m_file.GetFilename().GetStringRef()));
  if (!binary) {
    LLDB_LOG_ERROR(log, binary.takeError(),
                   "Failed to create binary for file ({1}): {0}", m_file);
    return false;
  }

  // Make sure we only handle COFF format.
  m_binary =
      llvm::unique_dyn_cast<llvm::object::COFFObjectFile>(std::move(*binary));
  if (!m_binary)
    return false;

  LLDB_LOG(log, "this = {0}, module = {1} ({2}), file = {3}, binary = {4}",
           this, GetModule().get(), GetModule()->GetSpecificationDescription(),
           m_file.GetPath(), m_binary.get());
  return true;
}

ObjectFilePECOFF::ObjectFilePECOFF(const lldb::ModuleSP &module_sp,
                                   DataBufferSP &data_sp,
                                   lldb::offset_t data_offset,
                                   const FileSpec *file,
                                   lldb::offset_t file_offset,
                                   lldb::offset_t length)
    : ObjectFile(module_sp, file, file_offset, length, data_sp, data_offset),
      m_dos_header(), m_coff_header(), m_sect_headers(),
      m_entry_point_address(), m_deps_filespec() {
  ::memset(&m_dos_header, 0, sizeof(m_dos_header));
  ::memset(&m_coff_header, 0, sizeof(m_coff_header));
}

ObjectFilePECOFF::ObjectFilePECOFF(const lldb::ModuleSP &module_sp,
                                   DataBufferSP &header_data_sp,
                                   const lldb::ProcessSP &process_sp,
                                   addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
      m_dos_header(), m_coff_header(), m_sect_headers(),
      m_entry_point_address(), m_deps_filespec() {
  ::memset(&m_dos_header, 0, sizeof(m_dos_header));
  ::memset(&m_coff_header, 0, sizeof(m_coff_header));
}

ObjectFilePECOFF::~ObjectFilePECOFF() = default;

bool ObjectFilePECOFF::ParseHeader() {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    m_sect_headers.clear();
    m_data.SetByteOrder(eByteOrderLittle);
    lldb::offset_t offset = 0;

    if (ParseDOSHeader(m_data, m_dos_header)) {
      offset = m_dos_header.e_lfanew;
      uint32_t pe_signature = m_data.GetU32(&offset);
      if (pe_signature != IMAGE_NT_SIGNATURE)
        return false;
      if (ParseCOFFHeader(m_data, &offset, m_coff_header)) {
        if (m_coff_header.hdrsize > 0)
          ParseCOFFOptionalHeader(&offset);
        ParseSectionHeaders(offset);
      }
      m_data.SetAddressByteSize(GetAddressByteSize());
      return true;
    }
  }
  return false;
}

bool ObjectFilePECOFF::SetLoadAddress(Target &target, addr_t value,
                                      bool value_is_offset) {
  bool changed = false;
  ModuleSP module_sp = GetModule();
  if (module_sp) {
    size_t num_loaded_sections = 0;
    SectionList *section_list = GetSectionList();
    if (section_list) {
      if (!value_is_offset) {
        value -= m_image_base;
      }

      const size_t num_sections = section_list->GetSize();
      size_t sect_idx = 0;

      for (sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
        // Iterate through the object file sections to find all of the sections
        // that have SHF_ALLOC in their flag bits.
        SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
        if (section_sp && !section_sp->IsThreadSpecific()) {
          if (target.GetSectionLoadList().SetSectionLoadAddress(
                  section_sp, section_sp->GetFileAddress() + value))
            ++num_loaded_sections;
        }
      }
      changed = num_loaded_sections > 0;
    }
  }
  return changed;
}

ByteOrder ObjectFilePECOFF::GetByteOrder() const { return eByteOrderLittle; }

bool ObjectFilePECOFF::IsExecutable() const {
  return (m_coff_header.flags & llvm::COFF::IMAGE_FILE_DLL) == 0;
}

uint32_t ObjectFilePECOFF::GetAddressByteSize() const {
  if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32_PLUS)
    return 8;
  else if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32)
    return 4;
  return 4;
}

// NeedsEndianSwap
//
// Return true if an endian swap needs to occur when extracting data from this
// file.
bool ObjectFilePECOFF::NeedsEndianSwap() const {
#if defined(__LITTLE_ENDIAN__)
  return false;
#else
  return true;
#endif
}
// ParseDOSHeader
bool ObjectFilePECOFF::ParseDOSHeader(DataExtractor &data,
                                      dos_header_t &dos_header) {
  bool success = false;
  lldb::offset_t offset = 0;
  success = data.ValidOffsetForDataOfSize(0, sizeof(dos_header));

  if (success) {
    dos_header.e_magic = data.GetU16(&offset); // Magic number
    success = dos_header.e_magic == IMAGE_DOS_SIGNATURE;

    if (success) {
      dos_header.e_cblp = data.GetU16(&offset); // Bytes on last page of file
      dos_header.e_cp = data.GetU16(&offset);   // Pages in file
      dos_header.e_crlc = data.GetU16(&offset); // Relocations
      dos_header.e_cparhdr =
          data.GetU16(&offset); // Size of header in paragraphs
      dos_header.e_minalloc =
          data.GetU16(&offset); // Minimum extra paragraphs needed
      dos_header.e_maxalloc =
          data.GetU16(&offset);               // Maximum extra paragraphs needed
      dos_header.e_ss = data.GetU16(&offset); // Initial (relative) SS value
      dos_header.e_sp = data.GetU16(&offset); // Initial SP value
      dos_header.e_csum = data.GetU16(&offset); // Checksum
      dos_header.e_ip = data.GetU16(&offset);   // Initial IP value
      dos_header.e_cs = data.GetU16(&offset);   // Initial (relative) CS value
      dos_header.e_lfarlc =
          data.GetU16(&offset); // File address of relocation table
      dos_header.e_ovno = data.GetU16(&offset); // Overlay number

      dos_header.e_res[0] = data.GetU16(&offset); // Reserved words
      dos_header.e_res[1] = data.GetU16(&offset); // Reserved words
      dos_header.e_res[2] = data.GetU16(&offset); // Reserved words
      dos_header.e_res[3] = data.GetU16(&offset); // Reserved words

      dos_header.e_oemid =
          data.GetU16(&offset); // OEM identifier (for e_oeminfo)
      dos_header.e_oeminfo =
          data.GetU16(&offset); // OEM information; e_oemid specific
      dos_header.e_res2[0] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[1] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[2] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[3] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[4] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[5] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[6] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[7] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[8] = data.GetU16(&offset); // Reserved words
      dos_header.e_res2[9] = data.GetU16(&offset); // Reserved words

      dos_header.e_lfanew =
          data.GetU32(&offset); // File address of new exe header
    }
  }
  if (!success)
    memset(&dos_header, 0, sizeof(dos_header));
  return success;
}

// ParserCOFFHeader
bool ObjectFilePECOFF::ParseCOFFHeader(DataExtractor &data,
                                       lldb::offset_t *offset_ptr,
                                       coff_header_t &coff_header) {
  bool success =
      data.ValidOffsetForDataOfSize(*offset_ptr, sizeof(coff_header));
  if (success) {
    coff_header.machine = data.GetU16(offset_ptr);
    coff_header.nsects = data.GetU16(offset_ptr);
    coff_header.modtime = data.GetU32(offset_ptr);
    coff_header.symoff = data.GetU32(offset_ptr);
    coff_header.nsyms = data.GetU32(offset_ptr);
    coff_header.hdrsize = data.GetU16(offset_ptr);
    coff_header.flags = data.GetU16(offset_ptr);
  }
  if (!success)
    memset(&coff_header, 0, sizeof(coff_header));
  return success;
}

bool ObjectFilePECOFF::ParseCOFFOptionalHeader(lldb::offset_t *offset_ptr) {
  bool success = false;
  const lldb::offset_t end_offset = *offset_ptr + m_coff_header.hdrsize;
  if (*offset_ptr < end_offset) {
    success = true;
    m_coff_header_opt.magic = m_data.GetU16(offset_ptr);
    m_coff_header_opt.major_linker_version = m_data.GetU8(offset_ptr);
    m_coff_header_opt.minor_linker_version = m_data.GetU8(offset_ptr);
    m_coff_header_opt.code_size = m_data.GetU32(offset_ptr);
    m_coff_header_opt.data_size = m_data.GetU32(offset_ptr);
    m_coff_header_opt.bss_size = m_data.GetU32(offset_ptr);
    m_coff_header_opt.entry = m_data.GetU32(offset_ptr);
    m_coff_header_opt.code_offset = m_data.GetU32(offset_ptr);

    const uint32_t addr_byte_size = GetAddressByteSize();

    if (*offset_ptr < end_offset) {
      if (m_coff_header_opt.magic == OPT_HEADER_MAGIC_PE32) {
        // PE32 only
        m_coff_header_opt.data_offset = m_data.GetU32(offset_ptr);
      } else
        m_coff_header_opt.data_offset = 0;

      if (*offset_ptr < end_offset) {
        m_coff_header_opt.image_base =
            m_data.GetMaxU64(offset_ptr, addr_byte_size);
        m_coff_header_opt.sect_alignment = m_data.GetU32(offset_ptr);
        m_coff_header_opt.file_alignment = m_data.GetU32(offset_ptr);
        m_coff_header_opt.major_os_system_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.minor_os_system_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.major_image_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.minor_image_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.major_subsystem_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.minor_subsystem_version = m_data.GetU16(offset_ptr);
        m_coff_header_opt.reserved1 = m_data.GetU32(offset_ptr);
        m_coff_header_opt.image_size = m_data.GetU32(offset_ptr);
        m_coff_header_opt.header_size = m_data.GetU32(offset_ptr);
        m_coff_header_opt.checksum = m_data.GetU32(offset_ptr);
        m_coff_header_opt.subsystem = m_data.GetU16(offset_ptr);
        m_coff_header_opt.dll_flags = m_data.GetU16(offset_ptr);
        m_coff_header_opt.stack_reserve_size =
            m_data.GetMaxU64(offset_ptr, addr_byte_size);
        m_coff_header_opt.stack_commit_size =
            m_data.GetMaxU64(offset_ptr, addr_byte_size);
        m_coff_header_opt.heap_reserve_size =
            m_data.GetMaxU64(offset_ptr, addr_byte_size);
        m_coff_header_opt.heap_commit_size =
            m_data.GetMaxU64(offset_ptr, addr_byte_size);
        m_coff_header_opt.loader_flags = m_data.GetU32(offset_ptr);
        uint32_t num_data_dir_entries = m_data.GetU32(offset_ptr);
        m_coff_header_opt.data_dirs.clear();
        m_coff_header_opt.data_dirs.resize(num_data_dir_entries);
        uint32_t i;
        for (i = 0; i < num_data_dir_entries; i++) {
          m_coff_header_opt.data_dirs[i].vmaddr = m_data.GetU32(offset_ptr);
          m_coff_header_opt.data_dirs[i].vmsize = m_data.GetU32(offset_ptr);
        }

        m_image_base = m_coff_header_opt.image_base;
      }
    }
  }
  // Make sure we are on track for section data which follows
  *offset_ptr = end_offset;
  return success;
}

uint32_t ObjectFilePECOFF::GetRVA(const Address &addr) const {
  return addr.GetFileAddress() - m_image_base;
}

Address ObjectFilePECOFF::GetAddress(uint32_t rva) {
  SectionList *sect_list = GetSectionList();
  if (!sect_list)
    return Address(GetFileAddress(rva));

  return Address(GetFileAddress(rva), sect_list);
}

lldb::addr_t ObjectFilePECOFF::GetFileAddress(uint32_t rva) const {
  return m_image_base + rva;
}

DataExtractor ObjectFilePECOFF::ReadImageData(uint32_t offset, size_t size) {
  if (!size)
    return {};

  if (m_data.ValidOffsetForDataOfSize(offset, size))
    return DataExtractor(m_data, offset, size);

  ProcessSP process_sp(m_process_wp.lock());
  DataExtractor data;
  if (process_sp) {
    auto data_up = std::make_unique<DataBufferHeap>(size, 0);
    Status readmem_error;
    size_t bytes_read =
        process_sp->ReadMemory(m_image_base + offset, data_up->GetBytes(),
                               data_up->GetByteSize(), readmem_error);
    if (bytes_read == size) {
      DataBufferSP buffer_sp(data_up.release());
      data.SetData(buffer_sp, 0, buffer_sp->GetByteSize());
    }
  }
  return data;
}

DataExtractor ObjectFilePECOFF::ReadImageDataByRVA(uint32_t rva, size_t size) {
  Address addr = GetAddress(rva);
  SectionSP sect = addr.GetSection();
  if (!sect)
    return {};
  rva = sect->GetFileOffset() + addr.GetOffset();

  return ReadImageData(rva, size);
}

// ParseSectionHeaders
bool ObjectFilePECOFF::ParseSectionHeaders(
    uint32_t section_header_data_offset) {
  const uint32_t nsects = m_coff_header.nsects;
  m_sect_headers.clear();

  if (nsects > 0) {
    const size_t section_header_byte_size = nsects * sizeof(section_header_t);
    DataExtractor section_header_data =
        ReadImageData(section_header_data_offset, section_header_byte_size);

    lldb::offset_t offset = 0;
    if (section_header_data.ValidOffsetForDataOfSize(
            offset, section_header_byte_size)) {
      m_sect_headers.resize(nsects);

      for (uint32_t idx = 0; idx < nsects; ++idx) {
        const void *name_data = section_header_data.GetData(&offset, 8);
        if (name_data) {
          memcpy(m_sect_headers[idx].name, name_data, 8);
          m_sect_headers[idx].vmsize = section_header_data.GetU32(&offset);
          m_sect_headers[idx].vmaddr = section_header_data.GetU32(&offset);
          m_sect_headers[idx].size = section_header_data.GetU32(&offset);
          m_sect_headers[idx].offset = section_header_data.GetU32(&offset);
          m_sect_headers[idx].reloff = section_header_data.GetU32(&offset);
          m_sect_headers[idx].lineoff = section_header_data.GetU32(&offset);
          m_sect_headers[idx].nreloc = section_header_data.GetU16(&offset);
          m_sect_headers[idx].nline = section_header_data.GetU16(&offset);
          m_sect_headers[idx].flags = section_header_data.GetU32(&offset);
        }
      }
    }
  }

  return !m_sect_headers.empty();
}

llvm::StringRef ObjectFilePECOFF::GetSectionName(const section_header_t &sect) {
  llvm::StringRef hdr_name(sect.name, llvm::array_lengthof(sect.name));
  hdr_name = hdr_name.split('\0').first;
  if (hdr_name.consume_front("/")) {
    lldb::offset_t stroff;
    if (!to_integer(hdr_name, stroff, 10))
      return "";
    lldb::offset_t string_file_offset =
        m_coff_header.symoff + (m_coff_header.nsyms * 18) + stroff;
    if (const char *name = m_data.GetCStr(&string_file_offset))
      return name;
    return "";
  }
  return hdr_name;
}

void ObjectFilePECOFF::ParseSymtab(Symtab &symtab) {
  SectionList *sect_list = GetSectionList();
  const uint32_t num_syms = m_coff_header.nsyms;
  if (m_file && num_syms > 0 && m_coff_header.symoff > 0) {
    const uint32_t symbol_size = 18;
    const size_t symbol_data_size = num_syms * symbol_size;
    // Include the 4-byte string table size at the end of the symbols
    DataExtractor symtab_data =
        ReadImageData(m_coff_header.symoff, symbol_data_size + 4);
    lldb::offset_t offset = symbol_data_size;
    const uint32_t strtab_size = symtab_data.GetU32(&offset);
    if (strtab_size > 0) {
      DataExtractor strtab_data = ReadImageData(
          m_coff_header.symoff + symbol_data_size, strtab_size);

      offset = 0;
      std::string symbol_name;
      Symbol *symbols = symtab.Resize(num_syms);
      for (uint32_t i = 0; i < num_syms; ++i) {
        coff_symbol_t symbol;
        const uint32_t symbol_offset = offset;
        const char *symbol_name_cstr = nullptr;
        // If the first 4 bytes of the symbol string are zero, then they
        // are followed by a 4-byte string table offset. Else these
        // 8 bytes contain the symbol name
        if (symtab_data.GetU32(&offset) == 0) {
          // Long string that doesn't fit into the symbol table name, so
          // now we must read the 4 byte string table offset
          uint32_t strtab_offset = symtab_data.GetU32(&offset);
          symbol_name_cstr = strtab_data.PeekCStr(strtab_offset);
          symbol_name.assign(symbol_name_cstr);
        } else {
          // Short string that fits into the symbol table name which is 8
          // bytes
          offset += sizeof(symbol.name) - 4; // Skip remaining
          symbol_name_cstr = symtab_data.PeekCStr(symbol_offset);
          if (symbol_name_cstr == nullptr)
            break;
          symbol_name.assign(symbol_name_cstr, sizeof(symbol.name));
        }
        symbol.value = symtab_data.GetU32(&offset);
        symbol.sect = symtab_data.GetU16(&offset);
        symbol.type = symtab_data.GetU16(&offset);
        symbol.storage = symtab_data.GetU8(&offset);
        symbol.naux = symtab_data.GetU8(&offset);
        symbols[i].GetMangled().SetValue(ConstString(symbol_name.c_str()));
        if ((int16_t)symbol.sect >= 1) {
          Address symbol_addr(sect_list->FindSectionByID(symbol.sect),
                              symbol.value);
          symbols[i].GetAddressRef() = symbol_addr;
          symbols[i].SetType(MapSymbolType(symbol.type));
        }

        if (symbol.naux > 0) {
          i += symbol.naux;
          offset += symbol.naux * symbol_size;
        }
      }
    }
  }

  // Read export header
  if (coff_data_dir_export_table < m_coff_header_opt.data_dirs.size() &&
      m_coff_header_opt.data_dirs[coff_data_dir_export_table].vmsize > 0 &&
      m_coff_header_opt.data_dirs[coff_data_dir_export_table].vmaddr > 0) {
    export_directory_entry export_table;
    uint32_t data_start =
        m_coff_header_opt.data_dirs[coff_data_dir_export_table].vmaddr;

    DataExtractor symtab_data = ReadImageDataByRVA(
        data_start, m_coff_header_opt.data_dirs[0].vmsize);
    lldb::offset_t offset = 0;

    // Read export_table header
    export_table.characteristics = symtab_data.GetU32(&offset);
    export_table.time_date_stamp = symtab_data.GetU32(&offset);
    export_table.major_version = symtab_data.GetU16(&offset);
    export_table.minor_version = symtab_data.GetU16(&offset);
    export_table.name = symtab_data.GetU32(&offset);
    export_table.base = symtab_data.GetU32(&offset);
    export_table.number_of_functions = symtab_data.GetU32(&offset);
    export_table.number_of_names = symtab_data.GetU32(&offset);
    export_table.address_of_functions = symtab_data.GetU32(&offset);
    export_table.address_of_names = symtab_data.GetU32(&offset);
    export_table.address_of_name_ordinals = symtab_data.GetU32(&offset);

    bool has_ordinal = export_table.address_of_name_ordinals != 0;

    lldb::offset_t name_offset = export_table.address_of_names - data_start;
    lldb::offset_t name_ordinal_offset =
        export_table.address_of_name_ordinals - data_start;

    Symbol *symbols = symtab.Resize(export_table.number_of_names);

    std::string symbol_name;

    // Read each export table entry
    for (size_t i = 0; i < export_table.number_of_names; ++i) {
      uint32_t name_ordinal =
          has_ordinal ? symtab_data.GetU16(&name_ordinal_offset) : i;
      uint32_t name_address = symtab_data.GetU32(&name_offset);

      const char *symbol_name_cstr =
          symtab_data.PeekCStr(name_address - data_start);
      symbol_name.assign(symbol_name_cstr);

      lldb::offset_t function_offset = export_table.address_of_functions -
                                        data_start +
                                        sizeof(uint32_t) * name_ordinal;
      uint32_t function_rva = symtab_data.GetU32(&function_offset);

      Address symbol_addr(m_coff_header_opt.image_base + function_rva,
                          sect_list);
      symbols[i].GetMangled().SetValue(ConstString(symbol_name.c_str()));
      symbols[i].GetAddressRef() = symbol_addr;
      symbols[i].SetType(lldb::eSymbolTypeCode);
      symbols[i].SetDebug(true);
    }
  }
}

std::unique_ptr<CallFrameInfo> ObjectFilePECOFF::CreateCallFrameInfo() {
  if (coff_data_dir_exception_table >= m_coff_header_opt.data_dirs.size())
    return {};

  data_directory data_dir_exception =
      m_coff_header_opt.data_dirs[coff_data_dir_exception_table];
  if (!data_dir_exception.vmaddr)
    return {};

  if (m_coff_header.machine != llvm::COFF::IMAGE_FILE_MACHINE_AMD64)
    return {};

  return std::make_unique<PECallFrameInfo>(*this, data_dir_exception.vmaddr,
                                           data_dir_exception.vmsize);
}

bool ObjectFilePECOFF::IsStripped() {
  // TODO: determine this for COFF
  return false;
}

SectionType ObjectFilePECOFF::GetSectionType(llvm::StringRef sect_name,
                                             const section_header_t &sect) {
  ConstString const_sect_name(sect_name);
  static ConstString g_code_sect_name(".code");
  static ConstString g_CODE_sect_name("CODE");
  static ConstString g_data_sect_name(".data");
  static ConstString g_DATA_sect_name("DATA");
  static ConstString g_bss_sect_name(".bss");
  static ConstString g_BSS_sect_name("BSS");

  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_CODE &&
      ((const_sect_name == g_code_sect_name) ||
       (const_sect_name == g_CODE_sect_name))) {
    return eSectionTypeCode;
  }
  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA &&
             ((const_sect_name == g_data_sect_name) ||
              (const_sect_name == g_DATA_sect_name))) {
    if (sect.size == 0 && sect.offset == 0)
      return eSectionTypeZeroFill;
    else
      return eSectionTypeData;
  }
  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA &&
             ((const_sect_name == g_bss_sect_name) ||
              (const_sect_name == g_BSS_sect_name))) {
    if (sect.size == 0)
      return eSectionTypeZeroFill;
    else
      return eSectionTypeData;
  }

  SectionType section_type =
      llvm::StringSwitch<SectionType>(sect_name)
          .Case(".debug", eSectionTypeDebug)
          .Case(".stabstr", eSectionTypeDataCString)
          .Case(".reloc", eSectionTypeOther)
          .Case(".debug_abbrev", eSectionTypeDWARFDebugAbbrev)
          .Case(".debug_aranges", eSectionTypeDWARFDebugAranges)
          .Case(".debug_frame", eSectionTypeDWARFDebugFrame)
          .Case(".debug_info", eSectionTypeDWARFDebugInfo)
          .Case(".debug_line", eSectionTypeDWARFDebugLine)
          .Case(".debug_loc", eSectionTypeDWARFDebugLoc)
          .Case(".debug_loclists", eSectionTypeDWARFDebugLocLists)
          .Case(".debug_macinfo", eSectionTypeDWARFDebugMacInfo)
          .Case(".debug_names", eSectionTypeDWARFDebugNames)
          .Case(".debug_pubnames", eSectionTypeDWARFDebugPubNames)
          .Case(".debug_pubtypes", eSectionTypeDWARFDebugPubTypes)
          .Case(".debug_ranges", eSectionTypeDWARFDebugRanges)
          .Case(".debug_str", eSectionTypeDWARFDebugStr)
          .Case(".debug_types", eSectionTypeDWARFDebugTypes)
          // .eh_frame can be truncated to 8 chars.
          .Cases(".eh_frame", ".eh_fram", eSectionTypeEHFrame)
          .Case(".gosymtab", eSectionTypeGoSymtab)
          .Default(eSectionTypeInvalid);
  if (section_type != eSectionTypeInvalid)
    return section_type;

  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_CODE)
    return eSectionTypeCode;
  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
    return eSectionTypeData;
  if (sect.flags & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
    if (sect.size == 0)
      return eSectionTypeZeroFill;
    else
      return eSectionTypeData;
  }
  return eSectionTypeOther;
}

void ObjectFilePECOFF::CreateSections(SectionList &unified_section_list) {
  if (m_sections_up)
    return;
  m_sections_up = std::make_unique<SectionList>();
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());

    SectionSP header_sp = std::make_shared<Section>(
        module_sp, this, ~user_id_t(0), ConstString("PECOFF header"),
        eSectionTypeOther, m_coff_header_opt.image_base,
        m_coff_header_opt.header_size,
        /*file_offset*/ 0, m_coff_header_opt.header_size,
        m_coff_header_opt.sect_alignment,
        /*flags*/ 0);
    header_sp->SetPermissions(ePermissionsReadable);
    m_sections_up->AddSection(header_sp);
    unified_section_list.AddSection(header_sp);

    const uint32_t nsects = m_sect_headers.size();
    ModuleSP module_sp(GetModule());
    for (uint32_t idx = 0; idx < nsects; ++idx) {
      llvm::StringRef sect_name = GetSectionName(m_sect_headers[idx]);
      ConstString const_sect_name(sect_name);
      SectionType section_type = GetSectionType(sect_name, m_sect_headers[idx]);

      SectionSP section_sp(new Section(
          module_sp,       // Module to which this section belongs
          this,            // Object file to which this section belongs
          idx + 1,         // Section ID is the 1 based section index.
          const_sect_name, // Name of this section
          section_type,
          m_coff_header_opt.image_base +
              m_sect_headers[idx].vmaddr, // File VM address == addresses as
                                          // they are found in the object file
          m_sect_headers[idx].vmsize,     // VM size in bytes of this section
          m_sect_headers[idx]
              .offset, // Offset to the data for this section in the file
          m_sect_headers[idx]
              .size, // Size in bytes of this section as found in the file
          m_coff_header_opt.sect_alignment, // Section alignment
          m_sect_headers[idx].flags));      // Flags for this section

      uint32_t permissions = 0;
      if (m_sect_headers[idx].flags & llvm::COFF::IMAGE_SCN_MEM_EXECUTE)
        permissions |= ePermissionsExecutable;
      if (m_sect_headers[idx].flags & llvm::COFF::IMAGE_SCN_MEM_READ)
        permissions |= ePermissionsReadable;
      if (m_sect_headers[idx].flags & llvm::COFF::IMAGE_SCN_MEM_WRITE)
        permissions |= ePermissionsWritable;
      section_sp->SetPermissions(permissions);

      m_sections_up->AddSection(section_sp);
      unified_section_list.AddSection(section_sp);
    }
  }
}

UUID ObjectFilePECOFF::GetUUID() {
  if (m_uuid.IsValid())
    return m_uuid;

  if (!CreateBinary())
    return UUID();

  m_uuid = GetCoffUUID(*m_binary);
  return m_uuid;
}

uint32_t ObjectFilePECOFF::ParseDependentModules() {
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return 0;

  std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
  if (m_deps_filespec)
    return m_deps_filespec->GetSize();

  // Cache coff binary if it is not done yet.
  if (!CreateBinary())
    return 0;

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_OBJECT));
  LLDB_LOG(log, "this = {0}, module = {1} ({2}), file = {3}, binary = {4}",
           this, GetModule().get(), GetModule()->GetSpecificationDescription(),
           m_file.GetPath(), m_binary.get());

  m_deps_filespec = FileSpecList();

  for (const auto &entry : m_binary->import_directories()) {
    llvm::StringRef dll_name;
    // Report a bogus entry.
    if (llvm::Error e = entry.getName(dll_name)) {
      LLDB_LOGF(log,
                "ObjectFilePECOFF::ParseDependentModules() - failed to get "
                "import directory entry name: %s",
                llvm::toString(std::move(e)).c_str());
      continue;
    }

    // At this moment we only have the base name of the DLL. The full path can
    // only be seen after the dynamic loading.  Our best guess is Try to get it
    // with the help of the object file's directory.
    llvm::SmallString<128> dll_fullpath;
    FileSpec dll_specs(dll_name);
    dll_specs.GetDirectory().SetString(m_file.GetDirectory().GetCString());

    if (!llvm::sys::fs::real_path(dll_specs.GetPath(), dll_fullpath))
      m_deps_filespec->EmplaceBack(dll_fullpath);
    else {
      // Known DLLs or DLL not found in the object file directory.
      m_deps_filespec->EmplaceBack(dll_name);
    }
  }
  return m_deps_filespec->GetSize();
}

uint32_t ObjectFilePECOFF::GetDependentModules(FileSpecList &files) {
  auto num_modules = ParseDependentModules();
  auto original_size = files.GetSize();

  for (unsigned i = 0; i < num_modules; ++i)
    files.AppendIfUnique(m_deps_filespec->GetFileSpecAtIndex(i));

  return files.GetSize() - original_size;
}

lldb_private::Address ObjectFilePECOFF::GetEntryPointAddress() {
  if (m_entry_point_address.IsValid())
    return m_entry_point_address;

  if (!ParseHeader() || !IsExecutable())
    return m_entry_point_address;

  SectionList *section_list = GetSectionList();
  addr_t file_addr = m_coff_header_opt.entry + m_coff_header_opt.image_base;

  if (!section_list)
    m_entry_point_address.SetOffset(file_addr);
  else
    m_entry_point_address.ResolveAddressUsingFileSections(file_addr,
                                                          section_list);
  return m_entry_point_address;
}

Address ObjectFilePECOFF::GetBaseAddress() {
  return Address(GetSectionList()->GetSectionAtIndex(0), 0);
}

// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
void ObjectFilePECOFF::Dump(Stream *s) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    s->Printf("%p: ", static_cast<void *>(this));
    s->Indent();
    s->PutCString("ObjectFilePECOFF");

    ArchSpec header_arch = GetArchitecture();

    *s << ", file = '" << m_file
       << "', arch = " << header_arch.GetArchitectureName() << "\n";

    SectionList *sections = GetSectionList();
    if (sections)
      sections->Dump(s->AsRawOstream(), s->GetIndentLevel(), nullptr, true,
                     UINT32_MAX);

    if (m_symtab_up)
      m_symtab_up->Dump(s, nullptr, eSortOrderNone);

    if (m_dos_header.e_magic)
      DumpDOSHeader(s, m_dos_header);
    if (m_coff_header.machine) {
      DumpCOFFHeader(s, m_coff_header);
      if (m_coff_header.hdrsize)
        DumpOptCOFFHeader(s, m_coff_header_opt);
    }
    s->EOL();
    DumpSectionHeaders(s);
    s->EOL();

    DumpDependentModules(s);
    s->EOL();
  }
}

// DumpDOSHeader
//
// Dump the MS-DOS header to the specified output stream
void ObjectFilePECOFF::DumpDOSHeader(Stream *s, const dos_header_t &header) {
  s->PutCString("MSDOS Header\n");
  s->Printf("  e_magic    = 0x%4.4x\n", header.e_magic);
  s->Printf("  e_cblp     = 0x%4.4x\n", header.e_cblp);
  s->Printf("  e_cp       = 0x%4.4x\n", header.e_cp);
  s->Printf("  e_crlc     = 0x%4.4x\n", header.e_crlc);
  s->Printf("  e_cparhdr  = 0x%4.4x\n", header.e_cparhdr);
  s->Printf("  e_minalloc = 0x%4.4x\n", header.e_minalloc);
  s->Printf("  e_maxalloc = 0x%4.4x\n", header.e_maxalloc);
  s->Printf("  e_ss       = 0x%4.4x\n", header.e_ss);
  s->Printf("  e_sp       = 0x%4.4x\n", header.e_sp);
  s->Printf("  e_csum     = 0x%4.4x\n", header.e_csum);
  s->Printf("  e_ip       = 0x%4.4x\n", header.e_ip);
  s->Printf("  e_cs       = 0x%4.4x\n", header.e_cs);
  s->Printf("  e_lfarlc   = 0x%4.4x\n", header.e_lfarlc);
  s->Printf("  e_ovno     = 0x%4.4x\n", header.e_ovno);
  s->Printf("  e_res[4]   = { 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x }\n",
            header.e_res[0], header.e_res[1], header.e_res[2], header.e_res[3]);
  s->Printf("  e_oemid    = 0x%4.4x\n", header.e_oemid);
  s->Printf("  e_oeminfo  = 0x%4.4x\n", header.e_oeminfo);
  s->Printf("  e_res2[10] = { 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, "
            "0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x }\n",
            header.e_res2[0], header.e_res2[1], header.e_res2[2],
            header.e_res2[3], header.e_res2[4], header.e_res2[5],
            header.e_res2[6], header.e_res2[7], header.e_res2[8],
            header.e_res2[9]);
  s->Printf("  e_lfanew   = 0x%8.8x\n", header.e_lfanew);
}

// DumpCOFFHeader
//
// Dump the COFF header to the specified output stream
void ObjectFilePECOFF::DumpCOFFHeader(Stream *s, const coff_header_t &header) {
  s->PutCString("COFF Header\n");
  s->Printf("  machine = 0x%4.4x\n", header.machine);
  s->Printf("  nsects  = 0x%4.4x\n", header.nsects);
  s->Printf("  modtime = 0x%8.8x\n", header.modtime);
  s->Printf("  symoff  = 0x%8.8x\n", header.symoff);
  s->Printf("  nsyms   = 0x%8.8x\n", header.nsyms);
  s->Printf("  hdrsize = 0x%4.4x\n", header.hdrsize);
}

// DumpOptCOFFHeader
//
// Dump the optional COFF header to the specified output stream
void ObjectFilePECOFF::DumpOptCOFFHeader(Stream *s,
                                         const coff_opt_header_t &header) {
  s->PutCString("Optional COFF Header\n");
  s->Printf("  magic                   = 0x%4.4x\n", header.magic);
  s->Printf("  major_linker_version    = 0x%2.2x\n",
            header.major_linker_version);
  s->Printf("  minor_linker_version    = 0x%2.2x\n",
            header.minor_linker_version);
  s->Printf("  code_size               = 0x%8.8x\n", header.code_size);
  s->Printf("  data_size               = 0x%8.8x\n", header.data_size);
  s->Printf("  bss_size                = 0x%8.8x\n", header.bss_size);
  s->Printf("  entry                   = 0x%8.8x\n", header.entry);
  s->Printf("  code_offset             = 0x%8.8x\n", header.code_offset);
  s->Printf("  data_offset             = 0x%8.8x\n", header.data_offset);
  s->Printf("  image_base              = 0x%16.16" PRIx64 "\n",
            header.image_base);
  s->Printf("  sect_alignment          = 0x%8.8x\n", header.sect_alignment);
  s->Printf("  file_alignment          = 0x%8.8x\n", header.file_alignment);
  s->Printf("  major_os_system_version = 0x%4.4x\n",
            header.major_os_system_version);
  s->Printf("  minor_os_system_version = 0x%4.4x\n",
            header.minor_os_system_version);
  s->Printf("  major_image_version     = 0x%4.4x\n",
            header.major_image_version);
  s->Printf("  minor_image_version     = 0x%4.4x\n",
            header.minor_image_version);
  s->Printf("  major_subsystem_version = 0x%4.4x\n",
            header.major_subsystem_version);
  s->Printf("  minor_subsystem_version = 0x%4.4x\n",
            header.minor_subsystem_version);
  s->Printf("  reserved1               = 0x%8.8x\n", header.reserved1);
  s->Printf("  image_size              = 0x%8.8x\n", header.image_size);
  s->Printf("  header_size             = 0x%8.8x\n", header.header_size);
  s->Printf("  checksum                = 0x%8.8x\n", header.checksum);
  s->Printf("  subsystem               = 0x%4.4x\n", header.subsystem);
  s->Printf("  dll_flags               = 0x%4.4x\n", header.dll_flags);
  s->Printf("  stack_reserve_size      = 0x%16.16" PRIx64 "\n",
            header.stack_reserve_size);
  s->Printf("  stack_commit_size       = 0x%16.16" PRIx64 "\n",
            header.stack_commit_size);
  s->Printf("  heap_reserve_size       = 0x%16.16" PRIx64 "\n",
            header.heap_reserve_size);
  s->Printf("  heap_commit_size        = 0x%16.16" PRIx64 "\n",
            header.heap_commit_size);
  s->Printf("  loader_flags            = 0x%8.8x\n", header.loader_flags);
  s->Printf("  num_data_dir_entries    = 0x%8.8x\n",
            (uint32_t)header.data_dirs.size());
  uint32_t i;
  for (i = 0; i < header.data_dirs.size(); i++) {
    s->Printf("  data_dirs[%2u] vmaddr = 0x%8.8x, vmsize = 0x%8.8x\n", i,
              header.data_dirs[i].vmaddr, header.data_dirs[i].vmsize);
  }
}
// DumpSectionHeader
//
// Dump a single ELF section header to the specified output stream
void ObjectFilePECOFF::DumpSectionHeader(Stream *s,
                                         const section_header_t &sh) {
  std::string name = std::string(GetSectionName(sh));
  s->Printf("%-16s 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%4.4x "
            "0x%4.4x 0x%8.8x\n",
            name.c_str(), sh.vmaddr, sh.vmsize, sh.offset, sh.size, sh.reloff,
            sh.lineoff, sh.nreloc, sh.nline, sh.flags);
}

// DumpSectionHeaders
//
// Dump all of the ELF section header to the specified output stream
void ObjectFilePECOFF::DumpSectionHeaders(Stream *s) {

  s->PutCString("Section Headers\n");
  s->PutCString("IDX  name             vm addr    vm size    file off   file "
                "size  reloc off  line off   nreloc nline  flags\n");
  s->PutCString("==== ---------------- ---------- ---------- ---------- "
                "---------- ---------- ---------- ------ ------ ----------\n");

  uint32_t idx = 0;
  SectionHeaderCollIter pos, end = m_sect_headers.end();

  for (pos = m_sect_headers.begin(); pos != end; ++pos, ++idx) {
    s->Printf("[%2u] ", idx);
    ObjectFilePECOFF::DumpSectionHeader(s, *pos);
  }
}

// DumpDependentModules
//
// Dump all of the dependent modules to the specified output stream
void ObjectFilePECOFF::DumpDependentModules(lldb_private::Stream *s) {
  auto num_modules = ParseDependentModules();
  if (num_modules > 0) {
    s->PutCString("Dependent Modules\n");
    for (unsigned i = 0; i < num_modules; ++i) {
      auto spec = m_deps_filespec->GetFileSpecAtIndex(i);
      s->Printf("  %s\n", spec.GetFilename().GetCString());
    }
  }
}

bool ObjectFilePECOFF::IsWindowsSubsystem() {
  switch (m_coff_header_opt.subsystem) {
  case llvm::COFF::IMAGE_SUBSYSTEM_NATIVE:
  case llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI:
  case llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI:
  case llvm::COFF::IMAGE_SUBSYSTEM_NATIVE_WINDOWS:
  case llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CE_GUI:
  case llvm::COFF::IMAGE_SUBSYSTEM_XBOX:
  case llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION:
    return true;
  default:
    return false;
  }
}

ArchSpec ObjectFilePECOFF::GetArchitecture() {
  uint16_t machine = m_coff_header.machine;
  switch (machine) {
  default:
    break;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
  case llvm::COFF::IMAGE_FILE_MACHINE_POWERPC:
  case llvm::COFF::IMAGE_FILE_MACHINE_POWERPCFP:
  case llvm::COFF::IMAGE_FILE_MACHINE_ARM:
  case llvm::COFF::IMAGE_FILE_MACHINE_ARMNT:
  case llvm::COFF::IMAGE_FILE_MACHINE_THUMB:
  case llvm::COFF::IMAGE_FILE_MACHINE_ARM64:
    ArchSpec arch;
    arch.SetArchitecture(eArchTypeCOFF, machine, LLDB_INVALID_CPUTYPE,
                         IsWindowsSubsystem() ? llvm::Triple::Win32
                                              : llvm::Triple::UnknownOS);
    return arch;
  }
  return ArchSpec();
}

ObjectFile::Type ObjectFilePECOFF::CalculateType() {
  if (m_coff_header.machine != 0) {
    if ((m_coff_header.flags & llvm::COFF::IMAGE_FILE_DLL) == 0)
      return eTypeExecutable;
    else
      return eTypeSharedLibrary;
  }
  return eTypeExecutable;
}

ObjectFile::Strata ObjectFilePECOFF::CalculateStrata() { return eStrataUser; }
