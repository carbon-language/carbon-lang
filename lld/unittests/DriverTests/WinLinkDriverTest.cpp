//===- lld/unittest/WinLinkDriverTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Windows link.exe driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"

#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/COFF.h"

#include <set>
#include <vector>

using namespace llvm;
using namespace lld;

namespace {

class WinLinkParserTest
    : public ParserTest<WinLinkDriver, PECOFFLinkingContext> {
protected:
  virtual const LinkingContext *linkingContext() { return &_context; }
};

TEST_F(WinLinkParserTest, Basic) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:console", "/out:a.exe",
                    "-entry:start", "a.obj", "b.obj", "c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _context.getSubsystem());
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_I386, _context.getMachineType());
  EXPECT_EQ("a.exe", _context.outputPath());
  EXPECT_EQ("_start", _context.entrySymbolName());
  EXPECT_EQ(4, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("c.obj", inputFile(2));
  EXPECT_TRUE(_context.getInputSearchPaths().empty());

  // Unspecified flags will have default values.
  EXPECT_FALSE(_context.isDll());
  EXPECT_EQ(6, _context.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _context.getMinOSVersion().minorVersion);
  EXPECT_EQ(0x400000U, _context.getBaseAddress());
  EXPECT_EQ(1024 * 1024U, _context.getStackReserve());
  EXPECT_EQ(4096U, _context.getStackCommit());
  EXPECT_EQ(4096U, _context.getSectionDefaultAlignment());
  EXPECT_FALSE(_context.allowRemainingUndefines());
  EXPECT_TRUE(_context.isNxCompat());
  EXPECT_FALSE(_context.getLargeAddressAware());
  EXPECT_TRUE(_context.getAllowBind());
  EXPECT_TRUE(_context.getAllowIsolation());
  EXPECT_FALSE(_context.getSwapRunFromCD());
  EXPECT_FALSE(_context.getSwapRunFromNet());
  EXPECT_TRUE(_context.getBaseRelocationEnabled());
  EXPECT_TRUE(_context.isTerminalServerAware());
  EXPECT_TRUE(_context.getDynamicBaseEnabled());
  EXPECT_TRUE(_context.getCreateManifest());
  EXPECT_EQ("a.exe.manifest", _context.getManifestOutputPath());
  EXPECT_EQ("", _context.getManifestDependency());
  EXPECT_FALSE(_context.getEmbedManifest());
  EXPECT_EQ(1, _context.getManifestId());
  EXPECT_EQ("'asInvoker'", _context.getManifestLevel());
  EXPECT_EQ("'false'", _context.getManifestUiAccess());
  EXPECT_TRUE(_context.deadStrip());
  EXPECT_FALSE(_context.logInputFiles());
}

TEST_F(WinLinkParserTest, StartsWithHyphen) {
  EXPECT_TRUE(
      parse("link.exe", "-subsystem:console", "-out:a.exe", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _context.getSubsystem());
  EXPECT_EQ("a.exe", _context.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, UppercaseOption) {
  EXPECT_TRUE(
      parse("link.exe", "/SUBSYSTEM:CONSOLE", "/OUT:a.exe", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _context.getSubsystem());
  EXPECT_EQ("a.exe", _context.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, Mllvm) {
  EXPECT_TRUE(parse("link.exe", "/mllvm:-debug", "a.obj", nullptr));
  const std::vector<const char *> &options = _context.llvmOptions();
  EXPECT_EQ(1U, options.size());
  EXPECT_STREQ("-debug", options[0]);
}

TEST_F(WinLinkParserTest, NoInputFiles) {
  EXPECT_FALSE(parse("link.exe", nullptr));
  EXPECT_EQ("No input files\n", errorMessage());
}

//
// Tests for implicit file extension interpolation.
//

TEST_F(WinLinkParserTest, NoFileExtension) {
  EXPECT_TRUE(parse("link.exe", "foo", "bar", nullptr));
  EXPECT_EQ("foo.exe", _context.outputPath());
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("foo.obj", inputFile(0));
  EXPECT_EQ("bar.obj", inputFile(1));
}

TEST_F(WinLinkParserTest, NonStandardFileExtension) {
  EXPECT_TRUE(parse("link.exe", "foo.o", nullptr));
  EXPECT_EQ("foo.exe", _context.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
}

TEST_F(WinLinkParserTest, Libpath) {
  EXPECT_TRUE(
      parse("link.exe", "/libpath:dir1", "/libpath:dir2", "a.obj", nullptr));
  const std::vector<StringRef> &paths = _context.getInputSearchPaths();
  EXPECT_EQ(2U, paths.size());
  EXPECT_EQ("dir1", paths[0]);
  EXPECT_EQ("dir2", paths[1]);
}

//
// Tests for input file order
//

TEST_F(WinLinkParserTest, InputOrder) {
  EXPECT_TRUE(parse("link.exe", "b.lib", "b.obj", "c.obj", "a.lib", "a.obj",
                    nullptr));
  EXPECT_EQ(6, inputFileCount());
  EXPECT_EQ("b.obj", inputFile(0));
  EXPECT_EQ("c.obj", inputFile(1));
  EXPECT_EQ("a.obj", inputFile(2));
  EXPECT_EQ("b.lib", inputFile(3));
  EXPECT_EQ("a.lib", inputFile(4));
}

//
// Tests for command line options that take values.
//

TEST_F(WinLinkParserTest, AlternateName) {
  EXPECT_TRUE(parse("link.exe", "/alternatename:sym1=sym2", "a.out", nullptr));
  EXPECT_EQ("sym1", _context.getAlternateName("sym2"));
  EXPECT_EQ("", _context.getAlternateName("foo"));
}

TEST_F(WinLinkParserTest, Export) {
  EXPECT_TRUE(parse("link.exe", "/export:foo", "a.out", nullptr));
  const std::set<PECOFFLinkingContext::ExportDesc> &exports =
      _context.getDllExports();
  EXPECT_EQ(1U, exports.size());
  auto it = exports.begin();
  EXPECT_EQ("_foo", it->name);
  EXPECT_EQ(-1, it->ordinal);
  EXPECT_FALSE(it->noname);
  EXPECT_FALSE(it->isData);
}

TEST_F(WinLinkParserTest, ExportWithOptions) {
  EXPECT_TRUE(parse("link.exe", "/export:foo,@8,noname,data",
                    "/export:bar,@10,data", "a.out", nullptr));
  const std::set<PECOFFLinkingContext::ExportDesc> &exports =
      _context.getDllExports();
  EXPECT_EQ(2U, exports.size());
  auto it = exports.begin();
  EXPECT_EQ("_bar", it->name);
  EXPECT_EQ(10, it->ordinal);
  EXPECT_FALSE(it->noname);
  EXPECT_TRUE(it->isData);
  ++it;
  EXPECT_EQ("_foo", it->name);
  EXPECT_EQ(8, it->ordinal);
  EXPECT_TRUE(it->noname);
  EXPECT_TRUE(it->isData);
}

TEST_F(WinLinkParserTest, ExportDuplicateExports) {
  EXPECT_TRUE(
      parse("link.exe", "/export:foo,@1", "/export:foo,@2", "a.out", nullptr));
  const std::set<PECOFFLinkingContext::ExportDesc> &exports =
      _context.getDllExports();
  EXPECT_EQ(1U, exports.size());
  auto it = exports.begin();
  EXPECT_EQ("_foo", it->name);
  EXPECT_EQ(1, it->ordinal);
}

TEST_F(WinLinkParserTest, ExportDuplicateOrdinals) {
  EXPECT_FALSE(
      parse("link.exe", "/export:foo,@1", "/export:bar,@1", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, ExportInvalid1) {
  EXPECT_FALSE(parse("link.exe", "/export:foo,@0", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, ExportInvalid2) {
  EXPECT_FALSE(parse("link.exe", "/export:foo,@65536", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, MachineX86) {
  EXPECT_TRUE(parse("link.exe", "/machine:x86", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_I386, _context.getMachineType());
}

TEST_F(WinLinkParserTest, MachineX64) {
  EXPECT_TRUE(parse("link.exe", "/machine:x64", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_AMD64, _context.getMachineType());
}

TEST_F(WinLinkParserTest, MachineArm) {
  EXPECT_FALSE(parse("link.exe", "/machine:arm", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage()).startswith(
      "Machine type other than x86/x64 is not supported"));
}

TEST_F(WinLinkParserTest, MajorImageVersion) {
  EXPECT_TRUE(parse("link.exe", "/version:7", "foo.o", nullptr));
  EXPECT_EQ(7, _context.getImageVersion().majorVersion);
  EXPECT_EQ(0, _context.getImageVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MajorMinorImageVersion) {
  EXPECT_TRUE(parse("link.exe", "/version:72.35", "foo.o", nullptr));
  EXPECT_EQ(72, _context.getImageVersion().majorVersion);
  EXPECT_EQ(35, _context.getImageVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorOSVersion) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:windows,3", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _context.getSubsystem());
  EXPECT_EQ(3, _context.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _context.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorMinorOSVersion) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:windows,3.1", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _context.getSubsystem());
  EXPECT_EQ(3, _context.getMinOSVersion().majorVersion);
  EXPECT_EQ(1, _context.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, Base) {
  EXPECT_TRUE(parse("link.exe", "/base:8388608", "a.obj", nullptr));
  EXPECT_EQ(0x800000U, _context.getBaseAddress());
}

TEST_F(WinLinkParserTest, InvalidBase) {
  EXPECT_FALSE(parse("link.exe", "/base:1234", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage())
                  .startswith("Base address have to be multiple of 64K"));
}

TEST_F(WinLinkParserTest, StackReserve) {
  EXPECT_TRUE(parse("link.exe", "/stack:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _context.getStackReserve());
  EXPECT_EQ(4096U, _context.getStackCommit());
}

TEST_F(WinLinkParserTest, StackReserveAndCommit) {
  EXPECT_TRUE(parse("link.exe", "/stack:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _context.getStackReserve());
  EXPECT_EQ(8192U, _context.getStackCommit());
}

TEST_F(WinLinkParserTest, InvalidStackSize) {
  EXPECT_FALSE(parse("link.exe", "/stack:8192,16384", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage()).startswith("Invalid stack size"));
}

TEST_F(WinLinkParserTest, HeapReserve) {
  EXPECT_TRUE(parse("link.exe", "/heap:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _context.getHeapReserve());
  EXPECT_EQ(4096U, _context.getHeapCommit());
}

TEST_F(WinLinkParserTest, HeapReserveAndCommit) {
  EXPECT_TRUE(parse("link.exe", "/heap:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _context.getHeapReserve());
  EXPECT_EQ(8192U, _context.getHeapCommit());
}

TEST_F(WinLinkParserTest, InvalidHeapSize) {
  EXPECT_FALSE(parse("link.exe", "/heap:8192,16384", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage()).startswith("Invalid heap size"));
}

TEST_F(WinLinkParserTest, SectionAlignment) {
  EXPECT_TRUE(parse("link.exe", "/align:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _context.getSectionDefaultAlignment());
}

TEST_F(WinLinkParserTest, InvalidAlignment) {
  EXPECT_FALSE(parse("link.exe", "/align:1000", "a.obj", nullptr));
  EXPECT_EQ("Section alignment must be a power of 2, but got 1000\n",
            errorMessage());
}

TEST_F(WinLinkParserTest, Include) {
  EXPECT_TRUE(parse("link.exe", "/include:foo", "a.out", nullptr));
  auto symbols = _context.initialUndefinedSymbols();
  EXPECT_FALSE(symbols.empty());
  EXPECT_EQ("foo", symbols[0]);
}

TEST_F(WinLinkParserTest, Merge) {
  EXPECT_TRUE(parse("link.exe", "/merge:.foo=.bar", "/merge:.bar=.baz",
                    "a.out", nullptr));
  EXPECT_EQ(".baz", _context.getOutputSectionName(".foo"));
  EXPECT_EQ(".baz", _context.getOutputSectionName(".bar"));
  EXPECT_EQ(".abc", _context.getOutputSectionName(".abc"));
}

TEST_F(WinLinkParserTest, Merge_Circular) {
  EXPECT_FALSE(parse("link.exe", "/merge:.foo=.bar", "/merge:.bar=.foo",
                     "a.out", nullptr));
}

//
// Tests for /section
//

namespace {
const uint32_t discardable = llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE;
const uint32_t not_cached = llvm::COFF::IMAGE_SCN_MEM_NOT_CACHED;
const uint32_t not_paged = llvm::COFF::IMAGE_SCN_MEM_NOT_PAGED;
const uint32_t shared = llvm::COFF::IMAGE_SCN_MEM_SHARED;
const uint32_t execute = llvm::COFF::IMAGE_SCN_MEM_EXECUTE;
const uint32_t read = llvm::COFF::IMAGE_SCN_MEM_READ;
const uint32_t write = llvm::COFF::IMAGE_SCN_MEM_WRITE;

#define TEST_SECTION(testname, arg, expect)                                    \
  TEST_F(WinLinkParserTest, testname) {                                        \
    EXPECT_TRUE(parse("link.exe", "/section:.text," arg, "a.obj", nullptr));   \
    EXPECT_EQ(expect, _context.getSectionAttributes(".text", execute | read)); \
  }

TEST_SECTION(SectionD, "d", execute | read | discardable);
TEST_SECTION(SectionE, "e", execute);
TEST_SECTION(SectionK, "k", execute | read | not_cached);
TEST_SECTION(SectionP, "p", execute | read | not_paged);
TEST_SECTION(SectionR, "r", read);
TEST_SECTION(SectionS, "s", execute | read | shared);
TEST_SECTION(SectionW, "w", write);

#undef TEST_SECTION

TEST_F(WinLinkParserTest, Section) {
  EXPECT_TRUE(parse("link.exe", "/section:.text,dekprsw",
                    "/section:.text,!dekprsw", "a.obj", nullptr));
  EXPECT_EQ(0U, _context.getSectionAttributes(".text", execute | read));
}

TEST_F(WinLinkParserTest, SectionNegate) {
  EXPECT_TRUE(parse("link.exe", "/section:.text,!e", "a.obj", nullptr));
  EXPECT_EQ(read, _context.getSectionAttributes(".text", execute | read));
}

TEST_F(WinLinkParserTest, SectionMultiple) {
  EXPECT_TRUE(parse("link.exe", "/section:.foo,e", "/section:.foo,rw",
                    "/section:.foo,!d", "a.obj", nullptr));
  uint32_t flags = execute | read | not_paged | discardable;
  uint32_t expected = execute | read | write | not_paged;
  EXPECT_EQ(expected, _context.getSectionAttributes(".foo", flags));
}

} // end anonymous namespace

//
// Tests for /defaultlib and /nodefaultlib.
//

TEST_F(WinLinkParserTest, DefaultLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "a.obj", nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("user32.lib", inputFile(1, 0));
  EXPECT_EQ("kernel32.lib", inputFile(1, 1));
}

TEST_F(WinLinkParserTest, DefaultLibDuplicates) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:user32.lib", "a.obj", nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("user32.lib", inputFile(1, 0));
}

TEST_F(WinLinkParserTest, NoDefaultLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/nodefaultlib:user32.lib", "a.obj",
                    nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("kernel32.lib", inputFile(1, 0));
}

TEST_F(WinLinkParserTest, NoDefaultLibAll) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/nodefaultlib", "a.obj", nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, DisallowLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/disallowlib:user32.lib", "a.obj",
                    nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("kernel32.lib", inputFile(1, 0));
}

//
// Tests for DLL.
//

TEST_F(WinLinkParserTest, NoEntry) {
  EXPECT_TRUE(parse("link.exe", "/noentry", "/dll", "a.obj", nullptr));
  EXPECT_TRUE(_context.isDll());
  EXPECT_EQ(0x10000000U, _context.getBaseAddress());
  EXPECT_EQ("", _context.entrySymbolName());
}

TEST_F(WinLinkParserTest, NoEntryError) {
  // /noentry without /dll is an error.
  EXPECT_FALSE(parse("link.exe", "/noentry", "a.obj", nullptr));
  EXPECT_EQ("/noentry must be specified with /dll\n", errorMessage());
}

//
// Tests for SEH.
//

TEST_F(WinLinkParserTest, SafeSEH) {
  EXPECT_TRUE(parse("link.exe", "/safeseh", "a.obj", nullptr));
  EXPECT_TRUE(_context.requireSEH());
  EXPECT_FALSE(_context.noSEH());
}

TEST_F(WinLinkParserTest, NoSafeSEH) {
  EXPECT_TRUE(parse("link.exe", "/safeseh:no", "a.obj", nullptr));
  EXPECT_FALSE(_context.requireSEH());
  EXPECT_TRUE(_context.noSEH());
}

//
// Tests for boolean flags.
//

TEST_F(WinLinkParserTest, Force) {
  EXPECT_TRUE(parse("link.exe", "/force", "a.obj", nullptr));
  EXPECT_TRUE(_context.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, ForceUnresolved) {
  EXPECT_TRUE(parse("link.exe", "/force:unresolved", "a.obj", nullptr));
  EXPECT_TRUE(_context.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, NoNxCompat) {
  EXPECT_TRUE(parse("link.exe", "/nxcompat:no", "a.obj", nullptr));
  EXPECT_FALSE(_context.isNxCompat());
}

TEST_F(WinLinkParserTest, LargeAddressAware) {
  EXPECT_TRUE(parse("link.exe", "/largeaddressaware", "a.obj", nullptr));
  EXPECT_TRUE(_context.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, NoLargeAddressAware) {
  EXPECT_TRUE(parse("link.exe", "/largeaddressaware:no", "a.obj", nullptr));
  EXPECT_FALSE(_context.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, AllowBind) {
  EXPECT_TRUE(parse("link.exe", "/allowbind", "a.obj", nullptr));
  EXPECT_TRUE(_context.getAllowBind());
}

TEST_F(WinLinkParserTest, NoAllowBind) {
  EXPECT_TRUE(parse("link.exe", "/allowbind:no", "a.obj", nullptr));
  EXPECT_FALSE(_context.getAllowBind());
}

TEST_F(WinLinkParserTest, AllowIsolation) {
  EXPECT_TRUE(parse("link.exe", "/allowisolation", "a.obj", nullptr));
  EXPECT_TRUE(_context.getAllowIsolation());
}

TEST_F(WinLinkParserTest, NoAllowIsolation) {
  EXPECT_TRUE(parse("link.exe", "/allowisolation:no", "a.obj", nullptr));
  EXPECT_FALSE(_context.getAllowIsolation());
}

TEST_F(WinLinkParserTest, SwapRunFromCD) {
  EXPECT_TRUE(parse("link.exe", "/swaprun:cd", "a.obj", nullptr));
  EXPECT_TRUE(_context.getSwapRunFromCD());
}

TEST_F(WinLinkParserTest, SwapRunFromNet) {
  EXPECT_TRUE(parse("link.exe", "/swaprun:net", "a.obj", nullptr));
  EXPECT_TRUE(_context.getSwapRunFromNet());
}

TEST_F(WinLinkParserTest, Debug) {
  EXPECT_TRUE(parse("link.exe", "/debug", "a.out", nullptr));
  EXPECT_FALSE(_context.deadStrip());
}

TEST_F(WinLinkParserTest, Fixed) {
  EXPECT_TRUE(parse("link.exe", "/fixed", "a.out", nullptr));
  EXPECT_FALSE(_context.getBaseRelocationEnabled());
  EXPECT_FALSE(_context.getDynamicBaseEnabled());
}

TEST_F(WinLinkParserTest, NoFixed) {
  EXPECT_TRUE(parse("link.exe", "/fixed:no", "a.out", nullptr));
  EXPECT_TRUE(_context.getBaseRelocationEnabled());
}

TEST_F(WinLinkParserTest, TerminalServerAware) {
  EXPECT_TRUE(parse("link.exe", "/tsaware", "a.out", nullptr));
  EXPECT_TRUE(_context.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, NoTerminalServerAware) {
  EXPECT_TRUE(parse("link.exe", "/tsaware:no", "a.out", nullptr));
  EXPECT_FALSE(_context.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, DynamicBase) {
  EXPECT_TRUE(parse("link.exe", "/dynamicbase", "a.out", nullptr));
  EXPECT_TRUE(_context.getDynamicBaseEnabled());
}

TEST_F(WinLinkParserTest, NoDynamicBase) {
  EXPECT_TRUE(parse("link.exe", "/dynamicbase:no", "a.out", nullptr));
  EXPECT_FALSE(_context.getDynamicBaseEnabled());
}

//
// Test for /failifmismatch
//

TEST_F(WinLinkParserTest, FailIfMismatch_Match) {
  EXPECT_TRUE(parse("link.exe", "/failifmismatch:foo=bar",
                    "/failifmismatch:foo=bar", "/failifmismatch:abc=def",
                    "a.out", nullptr));
}

TEST_F(WinLinkParserTest, FailIfMismatch_Mismatch) {
  EXPECT_FALSE(parse("link.exe", "/failifmismatch:foo=bar",
                     "/failifmismatch:foo=baz", "a.out", nullptr));
}

//
// Tests for /manifest, /manifestuac, /manifestfile, and /manifestdependency.
//
TEST_F(WinLinkParserTest, Manifest_Default) {
  EXPECT_TRUE(parse("link.exe", "/manifest", "a.out", nullptr));
  EXPECT_TRUE(_context.getCreateManifest());
  EXPECT_FALSE(_context.getEmbedManifest());
  EXPECT_EQ(1, _context.getManifestId());
  EXPECT_EQ("'asInvoker'", _context.getManifestLevel());
  EXPECT_EQ("'false'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifest_No) {
  EXPECT_TRUE(parse("link.exe", "/manifest:no", "a.out", nullptr));
  EXPECT_FALSE(_context.getCreateManifest());
}

TEST_F(WinLinkParserTest, Manifest_Embed) {
  EXPECT_TRUE(parse("link.exe", "/manifest:embed", "a.out", nullptr));
  EXPECT_TRUE(_context.getCreateManifest());
  EXPECT_TRUE(_context.getEmbedManifest());
  EXPECT_EQ(1, _context.getManifestId());
  EXPECT_EQ("'asInvoker'", _context.getManifestLevel());
  EXPECT_EQ("'false'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifest_Embed_ID42) {
  EXPECT_TRUE(parse("link.exe", "/manifest:embed,id=42", "a.out", nullptr));
  EXPECT_TRUE(_context.getCreateManifest());
  EXPECT_TRUE(_context.getEmbedManifest());
  EXPECT_EQ(42, _context.getManifestId());
  EXPECT_EQ("'asInvoker'", _context.getManifestLevel());
  EXPECT_EQ("'false'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestuac_Level) {
  EXPECT_TRUE(parse("link.exe", "/manifestuac:level='requireAdministrator'",
                    "a.out", nullptr));
  EXPECT_EQ("'requireAdministrator'", _context.getManifestLevel());
  EXPECT_EQ("'false'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestuac_UiAccess) {
  EXPECT_TRUE(parse("link.exe", "/manifestuac:uiAccess='true'", "a.out", nullptr));
  EXPECT_EQ("'asInvoker'", _context.getManifestLevel());
  EXPECT_EQ("'true'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestuac_LevelAndUiAccess) {
  EXPECT_TRUE(parse("link.exe",
                    "/manifestuac:level='requireAdministrator' uiAccess='true'",
                    "a.out", nullptr));
  EXPECT_EQ("'requireAdministrator'", _context.getManifestLevel());
  EXPECT_EQ("'true'", _context.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestfile) {
  EXPECT_TRUE(parse("link.exe", "/manifestfile:bar.manifest",
                    "a.out", nullptr));
  EXPECT_EQ("bar.manifest", _context.getManifestOutputPath());
}

TEST_F(WinLinkParserTest, Manifestdependency) {
  EXPECT_TRUE(parse("link.exe", "/manifestdependency:foo bar", "a.out",
                    nullptr));
  EXPECT_EQ("foo bar", _context.getManifestDependency());
}

//
// Test for command line flags that are ignored.
//

TEST_F(WinLinkParserTest, Ignore) {
  // There are some no-op command line options that are recognized for
  // compatibility with link.exe.
  EXPECT_TRUE(parse("link.exe", "/nologo", "/errorreport:prompt",
                    "/incremental", "/incremental:no", "/delay:unload",
                    "/disallowlib:foo", "/delayload:user32", "/pdb:foo",
                    "/pdbaltpath:bar", "/verbose", "/verbose:icf", "/wx",
                    "/wx:no", "/tlbid:1", "/tlbout:foo", "/idlout:foo",
                    "/ignoreidl", "/implib:foo", "/safeseh", "/safeseh:no",
                    "/functionpadmin", "a.obj", nullptr));
  EXPECT_EQ("", errorMessage());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

//
// Test for "--"
//

TEST_F(WinLinkParserTest, DashDash) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:console", "/out:a.exe", "a.obj",
                    "--", "b.obj", "-c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _context.getSubsystem());
  EXPECT_EQ("a.exe", _context.outputPath());
  EXPECT_EQ(4, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("-c.obj", inputFile(2));
}

//
// Tests for entry symbol name.
//

TEST_F(WinLinkParserTest, DefEntryNameConsole) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:console", "a.obj", nullptr));
  EXPECT_EQ("_mainCRTStartup", _context.entrySymbolName());
}

TEST_F(WinLinkParserTest, DefEntryNameWindows) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:windows", "a.obj", nullptr));
  EXPECT_EQ("_WinMainCRTStartup", _context.entrySymbolName());
}

} // end anonymous namespace
