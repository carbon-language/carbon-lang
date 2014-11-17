#include <stdio.h>
#include <stdint.h>
#include <mach-o/loader.h>
#include <mach-o/compact_unwind_encoding.h>
#include <mach/machine.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <inttypes.h>


// A quick sketch of a program which can parse the compact unwind info
// used on Darwin systems for exception handling.  The output of
// unwinddump will be more authoritative/reliable but this program
// can dump at least the UNWIND_X86_64_MODE_RBP_FRAME format entries
// correctly.

struct baton
{
    cpu_type_t cputype;

    uint8_t *mach_header_start;      // pointer into this program's address space
    uint8_t *compact_unwind_start;   // pointer into this program's address space

    int addr_size;                   // 4 or 8 bytes, the size of addresses in this file

    uint64_t text_segment_vmaddr;
    uint64_t eh_section_file_address; // the file address of the __TEXT,__eh_frame section

    uint8_t *lsda_array_start;       // for the currently-being-processed first-level index
    uint8_t *lsda_array_end;         // the lsda_array_start for the NEXT first-level index

    struct unwind_info_section_header unwind_header;
    struct unwind_info_section_header_index_entry first_level_index_entry;
    struct unwind_info_compressed_second_level_page_header compressed_second_level_page_header;
    struct unwind_info_regular_second_level_page_header regular_second_level_page_header;
};


// step through the load commands in a thin mach-o binary,
// find the cputype and the start of the __TEXT,__unwind_info
// section, return a pointer to that section or NULL if not found.

void
scan_macho_load_commands (struct baton *baton)
{
    baton->compact_unwind_start = 0;

    uint32_t *magic = (uint32_t *) baton->mach_header_start;

    if (*magic != MH_MAGIC && *magic != MH_MAGIC_64)
    {
        printf ("Unexpected magic number 0x%x in header, exiting.", *magic);
        exit (1);
    }

    bool is_64bit = false;
    if (*magic == MH_MAGIC_64)
        is_64bit = true;

    uint8_t *offset = baton->mach_header_start;

    struct mach_header mh;
    memcpy (&mh, offset, sizeof (struct mach_header));
    if (is_64bit)
        offset += sizeof (struct mach_header_64);
    else
        offset += sizeof (struct mach_header);

    if (is_64bit)
        baton->addr_size = 8;
    else
        baton->addr_size = 4;

    baton->cputype = mh.cputype;

    uint32_t cur_cmd = 0;
    while (cur_cmd < mh.ncmds && (offset - baton->mach_header_start) < mh.sizeofcmds)
    {
        struct load_command lc;
        uint32_t *lc_cmd = (uint32_t *) offset;
        uint32_t *lc_cmdsize = (uint32_t *) offset + 1;
        uint8_t *start_of_this_load_cmd = offset;
        
        char segment_name[17];
        segment_name[0] = '\0';
        uint32_t nsects = 0;
        uint64_t segment_offset = 0;
        uint64_t segment_vmaddr = 0;

        if (*lc_cmd == LC_SEGMENT_64)
        {
            struct segment_command_64 seg;
            memcpy (&seg, offset, sizeof (struct segment_command_64));
            memcpy (&segment_name, &seg.segname, 16);
            segment_name[16] = '\0';
            nsects = seg.nsects;
            segment_offset = seg.fileoff;
            segment_vmaddr = seg.vmaddr;
            offset += sizeof (struct segment_command_64);
        }

        if (*lc_cmd == LC_SEGMENT)
        {
            struct segment_command seg;
            memcpy (&seg, offset, sizeof (struct segment_command));
            memcpy (&segment_name, &seg.segname, 16);
            segment_name[16] = '\0';
            nsects = seg.nsects;
            segment_offset = seg.fileoff;
            segment_vmaddr = seg.vmaddr;
            offset += sizeof (struct segment_command);
        }

        if (nsects != 0 && segment_name[0] != '\0' && strcmp (segment_name, "__TEXT") == 0)
        {
            baton->text_segment_vmaddr = segment_vmaddr;

            uint32_t current_sect = 0;
            while (current_sect < nsects && (offset - start_of_this_load_cmd) < *lc_cmdsize)
            {
                char sect_name[17];
                memcpy (&sect_name, offset, 16);
                sect_name[16] = '\0';
                if (strcmp (sect_name, "__unwind_info") == 0)
                {
                    if (is_64bit)
                    {
                        struct section_64 sect;
                        memcpy (&sect, offset, sizeof (struct section_64));
                        baton->compact_unwind_start = baton->mach_header_start + sect.offset;
                    }
                    else
                    {
                        struct section sect;
                        memcpy (&sect, offset, sizeof (struct section));
                        baton->compact_unwind_start = baton->mach_header_start + sect.offset;
                    }
                }
                if (strcmp (sect_name, "__eh_frame") == 0)
                {
                    if (is_64bit)
                    {
                        struct section_64 sect;
                        memcpy (&sect, offset, sizeof (struct section_64));
                        baton->eh_section_file_address = sect.addr;
                    }
                    else
                    {
                        struct section sect;
                        memcpy (&sect, offset, sizeof (struct section));
                        baton->eh_section_file_address = sect.addr;
                    }
                }

                if (is_64bit)
                {
                    offset += sizeof (struct section_64);
                }
                else
                {
                    offset += sizeof (struct section);
                }
            }

            return;
        }

        offset = start_of_this_load_cmd + *lc_cmdsize;
        cur_cmd++;
    }
}

void
print_encoding_x86_64 (struct baton baton, uint32_t encoding)
{
    int mode = encoding & UNWIND_X86_64_MODE_MASK;
    switch (mode)
    {
        case UNWIND_X86_64_MODE_RBP_FRAME:
        {
            printf (" - frame func: CFA is rbp+%d ", 16);
            printf (" rip=[CFA-8] rbp=[CFA-16]");
            uint32_t saved_registers_offset = (encoding & UNWIND_X86_64_RBP_FRAME_OFFSET) >> (__builtin_ctz (UNWIND_X86_64_RBP_FRAME_OFFSET));

            uint32_t saved_registers_locations = (encoding & UNWIND_X86_64_RBP_FRAME_REGISTERS) >> (__builtin_ctz (UNWIND_X86_64_RBP_FRAME_REGISTERS));


            saved_registers_offset += 2;

            for (int i = 0; i < 5; i++)
            {
                switch (saved_registers_locations & 0x7)
                {
                    case UNWIND_X86_64_REG_NONE:
                        break;
                    case UNWIND_X86_64_REG_RBX:
                        printf (" rbx=[CFA-%d]", saved_registers_offset * 8);
                        break;
                    case UNWIND_X86_64_REG_R12:
                        printf (" r12=[CFA-%d]", saved_registers_offset * 8);
                        break;
                    case UNWIND_X86_64_REG_R13:
                        printf (" r13=[CFA-%d]", saved_registers_offset * 8);
                        break;
                    case UNWIND_X86_64_REG_R14:
                        printf (" r14=[CFA-%d]", saved_registers_offset * 8);
                        break;
                    case UNWIND_X86_64_REG_R15:
                        printf (" r15=[CFA-%d]", saved_registers_offset * 8);
                        break;
                }
                saved_registers_offset--;
                saved_registers_locations >>= 3;
            }
        }
        break;

        case UNWIND_X86_64_MODE_STACK_IND:
        {
            printf (" UNWIND_X86_64_MODE_STACK_IND not yet supported\n");
            break;
        }
        case UNWIND_X86_64_MODE_STACK_IMMD:
        {
            printf (" UNWIND_X86_64_MODE_STACK_IND not yet supported\n");
            break;

            // FIXME not getting the rbp register saves out of the register permutation yet

            uint32_t stack_size = (encoding & UNWIND_X86_64_FRAMELESS_STACK_SIZE) >> (__builtin_ctz (UNWIND_X86_64_FRAMELESS_STACK_SIZE));
            uint32_t stack_adjust = (encoding & UNWIND_X86_FRAMELESS_STACK_ADJUST) >> (__builtin_ctz (UNWIND_X86_FRAMELESS_STACK_ADJUST));
            uint32_t register_count = (encoding & UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT) >> (__builtin_ctz (UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT));
            uint32_t permutation = (encoding & UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION) >> (__builtin_ctz (UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION));
            
//            printf (" frameless function: stack size %d, stack adjust %d, register count %d ", stack_size * 8, stack_adjust * 8, register_count);
            if ((encoding & UNWIND_X86_64_MODE_MASK) == UNWIND_X86_64_MODE_STACK_IND)
            {
                printf (" UNWIND_X86_64_MODE_STACK_IND not handled ");
                // stack size is too large to store in UNWIND_X86_64_FRAMELESS_STACK_SIZE; instead 
                // stack_size is an offset into the function's instruction stream to the 32-bit literal
                // value in a "sub $xxx, %rsp" instruction.
                return;
            }

            if (register_count == 0)
            {
                printf (" no registers saved");
            }
            else
            {
                int permunreg[6];
                switch (register_count) 
                {
                    case 6:
                        permunreg[0] = permutation/120;
                        permutation -= (permunreg[0]*120);
                        permunreg[1] = permutation/24;
                        permutation -= (permunreg[1]*24);
                        permunreg[2] = permutation/6;
                        permutation -= (permunreg[2]*6);
                        permunreg[3] = permutation/2;
                        permutation -= (permunreg[3]*2);
                        permunreg[4] = permutation;
                        permunreg[5] = 0;
                        break;
                    case 5:
                        permunreg[0] = permutation/120;
                        permutation -= (permunreg[0]*120);
                        permunreg[1] = permutation/24;
                        permutation -= (permunreg[1]*24);
                        permunreg[2] = permutation/6;
                        permutation -= (permunreg[2]*6);
                        permunreg[3] = permutation/2;
                        permutation -= (permunreg[3]*2);
                        permunreg[4] = permutation;
                        break;
                    case 4:
                        permunreg[0] = permutation/60;
                        permutation -= (permunreg[0]*60);
                        permunreg[1] = permutation/12;
                        permutation -= (permunreg[1]*12);
                        permunreg[2] = permutation/3;
                        permutation -= (permunreg[2]*3);
                        permunreg[3] = permutation;
                        break;
                    case 3:
                        permunreg[0] = permutation/20;
                        permutation -= (permunreg[0]*20);
                        permunreg[1] = permutation/4;
                        permutation -= (permunreg[1]*4);
                        permunreg[2] = permutation;
                        break;
                    case 2:
                        permunreg[0] = permutation/5;
                        permutation -= (permunreg[0]*5);
                        permunreg[1] = permutation;
                        break;
                    case 1:
                        permunreg[0] = permutation;
                        break;
                }
                
                int registers[6];
                bool used[7] = { false, false, false, false, false, false, false };
                for (int i = 0; i < register_count; i++)
                {
                    int renum = 0;
                    for (int j = 0; j < 7; j++)
                    {
                        if (used[j] == false)
                        {
                            if (renum == permunreg[i])
                            {
                                registers[i] = j;
                                used[j] = true;
                                break;
                            }
                            renum++;
                        }
                    }
                }


                printf ("CFA is rsp+%d ", stack_size * 8);

                uint32_t saved_registers_offset = 1;
                printf (" rip=[CFA-%d]", saved_registers_offset * 8);
                saved_registers_offset++;

                for (int i = (sizeof (registers) / sizeof (int)) - 1; i >= 0; i--)
                {
                    switch (registers[i])
                    {
                        case UNWIND_X86_64_REG_NONE:
                            break;
                        case UNWIND_X86_64_REG_RBX:
                            printf (" rbx=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R12:
                            printf (" r12=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R13:
                            printf (" r13=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R14:
                            printf (" r14=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_R15:
                            printf (" r15=[CFA-%d]", saved_registers_offset * 8);
                            break;
                        case UNWIND_X86_64_REG_RBP:
                            printf (" rbp=[CFA-%d]", saved_registers_offset * 8);
                            break;
                    }
                    saved_registers_offset++;
                }

            }

        }
        break;

        case UNWIND_X86_64_MODE_DWARF:
        {
            uint32_t dwarf_offset = encoding & UNWIND_X86_DWARF_SECTION_OFFSET;
            printf (" use DWARF unwind instructions: FDE at offset %d (file address 0x%" PRIx64 ")\n",
                    dwarf_offset, dwarf_offset + baton.eh_section_file_address);
        }
        break;

    }
}

void print_encoding (struct baton baton, uint32_t encoding)
{

    if (baton.cputype == CPU_TYPE_X86_64)
    {
        print_encoding_x86_64 (baton, encoding);
    }
    else
    {
        printf (" -- unsupported encoding arch -- ");
    }
}

void
print_function_encoding (struct baton baton, uint32_t idx, uint32_t encoding, uint32_t entry_encoding_index, uint32_t entry_func_offset)
{

    char *entry_encoding_index_str = "";
    if (entry_encoding_index != (uint32_t) -1)
    {
        asprintf (&entry_encoding_index_str, ", encoding #%d", entry_encoding_index);
    }
    else
    {
        asprintf (&entry_encoding_index_str, "");
    }
    printf ("    func [%d] offset %d (file addr 0x%" PRIx64 ")%s - 0x%x", 
            idx, entry_func_offset, 
            entry_func_offset + baton.first_level_index_entry.functionOffset + baton.text_segment_vmaddr, 
            entry_encoding_index_str, 
            encoding);

    print_encoding (baton, encoding);

    bool has_lsda = encoding & UNWIND_HAS_LSDA;

    if (has_lsda)
    {
        uint32_t func_offset = entry_func_offset + baton.first_level_index_entry.functionOffset;

        uint32_t lsda_offset = 0;

        uint32_t low = 0;
        uint32_t high = (baton.lsda_array_end - baton.lsda_array_start) / sizeof (struct unwind_info_section_header_lsda_index_entry);

        while (low < high)
        {
            uint32_t mid = (low + high) / 2;

            uint8_t *mid_lsda_entry_addr = (baton.lsda_array_start + (mid * sizeof (struct unwind_info_section_header_lsda_index_entry)));
            struct unwind_info_section_header_lsda_index_entry mid_lsda_entry;
            memcpy (&mid_lsda_entry, mid_lsda_entry_addr, sizeof (struct unwind_info_section_header_lsda_index_entry));
            if (mid_lsda_entry.functionOffset == func_offset)
            {
                lsda_offset = mid_lsda_entry.lsdaOffset;
                break;
            }
            else if (mid_lsda_entry.functionOffset < func_offset)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        printf (", LSDA offset %d", lsda_offset);
    }

    uint32_t pers_idx = (encoding & UNWIND_PERSONALITY_MASK) >> (__builtin_ctz(UNWIND_PERSONALITY_MASK));
    if (pers_idx != 0)
    {
        pers_idx--;  // Change 1-based to 0-based index
        uint32_t pers_delta = *((uint32_t*) (baton.compact_unwind_start + baton.unwind_header.personalityArraySectionOffset + pers_idx * 4));

        uint8_t **personality_addr = (uint8_t **) (baton.mach_header_start + pers_delta);
        void *personality = *personality_addr;
        printf (", personality func addr @ offset %d", pers_delta);
//            printf (", personality %p", personality);
    }

    printf ("\n");
}

void
print_second_level_index_regular (struct baton baton)
{
    uint8_t *page_entries = baton.compact_unwind_start + baton.first_level_index_entry.secondLevelPagesSectionOffset + baton.regular_second_level_page_header.entryPageOffset;
    uint8_t entries_count =  baton.regular_second_level_page_header.entryCount;

    uint8_t *offset = page_entries;
    uint8_t idx = 0;

    for (uint32_t idx = 0; idx < entries_count; idx++, offset += 8)
    {
        uint32_t func_offset = *((uint32_t *) (offset));
        uint32_t encoding = *((uint32_t *) (offset + 4)); 
        print_function_encoding (baton, idx, encoding, (uint32_t) -1, func_offset);
    }
}

void
print_second_level_index_compressed (struct baton baton)
{
    uint8_t *this_index = baton.compact_unwind_start + baton.first_level_index_entry.secondLevelPagesSectionOffset;
    uint8_t *start_of_entries = this_index + baton.compressed_second_level_page_header.entryPageOffset;
    uint8_t *offset = start_of_entries;
    for (uint16_t idx = 0; idx < baton.compressed_second_level_page_header.entryCount; idx++)
    {
        uint32_t entry = *((uint32_t*) offset);
        offset += 4;
        uint32_t encoding;

        uint32_t entry_encoding_index = UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX (entry);
        uint32_t entry_func_offset = UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET (entry);

        if (entry_encoding_index < baton.unwind_header.commonEncodingsArrayCount)
        {
            // encoding is in common table in section header
            encoding = *((uint32_t*) (baton.compact_unwind_start + baton.unwind_header.commonEncodingsArraySectionOffset + (entry_encoding_index * sizeof (uint32_t))));
        }
        else
        {
            // encoding is in page specific table
            uint32_t page_encoding_index = entry_encoding_index - baton.unwind_header.commonEncodingsArrayCount;
            encoding = *((uint32_t*) (this_index + baton.compressed_second_level_page_header.encodingsPageOffset + (page_encoding_index * sizeof (uint32_t))));
        }


        print_function_encoding (baton, idx, encoding, entry_encoding_index, entry_func_offset);
    }
}

void
print_second_level_index (struct baton baton)
{
    uint8_t *index_start = baton.compact_unwind_start + baton.first_level_index_entry.secondLevelPagesSectionOffset;

    if ((*(uint32_t*) index_start) == UNWIND_SECOND_LEVEL_REGULAR)
    {
        struct unwind_info_regular_second_level_page_header header;
        memcpy (&header, index_start, sizeof (struct unwind_info_regular_second_level_page_header));
        printf ("  UNWIND_SECOND_LEVEL_REGULAR entryPageOffset %d, entryCount %d\n", header.entryPageOffset, header.entryCount);
        baton.regular_second_level_page_header = header;
        print_second_level_index_regular (baton);
    }

    if ((*(uint32_t*) index_start) == UNWIND_SECOND_LEVEL_COMPRESSED)
    {
        struct unwind_info_compressed_second_level_page_header header;
        memcpy (&header, index_start, sizeof (struct unwind_info_compressed_second_level_page_header));
        printf ("  UNWIND_SECOND_LEVEL_COMPRESSED entryPageOffset %d, entryCount %d, encodingsPageOffset %d, encodingsCount %d\n", header.entryPageOffset, header.entryCount, header.encodingsPageOffset, header.encodingsCount);
        baton.compressed_second_level_page_header = header;
        print_second_level_index_compressed (baton);
    }
}


void
print_index_sections (struct baton baton)
{    
    uint8_t *index_section_offset = baton.compact_unwind_start + baton.unwind_header.indexSectionOffset;
    uint32_t index_count = baton.unwind_header.indexCount;

    uint32_t cur_idx = 0;

    uint8_t *offset = index_section_offset;
    while (cur_idx < index_count)
    {
        struct unwind_info_section_header_index_entry index_entry;
        memcpy (&index_entry, offset, sizeof (struct unwind_info_section_header_index_entry));
        printf ("index section #%d: functionOffset %d, secondLevelPagesSectionOffset %d, lsdaIndexArraySectionOffset %d\n", cur_idx, index_entry.functionOffset, index_entry.secondLevelPagesSectionOffset, index_entry.lsdaIndexArraySectionOffset);

        // secondLevelPagesSectionOffset == 0 means this is a sentinel entry
        if (index_entry.secondLevelPagesSectionOffset != 0)
        {
            struct unwind_info_section_header_index_entry next_index_entry;
            memcpy (&next_index_entry, offset + sizeof (struct unwind_info_section_header_index_entry), sizeof (struct unwind_info_section_header_index_entry));

            baton.lsda_array_start = baton.compact_unwind_start + index_entry.lsdaIndexArraySectionOffset;
            baton.lsda_array_end = baton.compact_unwind_start + next_index_entry.lsdaIndexArraySectionOffset;

            uint8_t *lsda_entry_offset = baton.lsda_array_start;
            uint32_t lsda_count = 0;
            while (lsda_entry_offset < baton.lsda_array_end)
            {
                struct unwind_info_section_header_lsda_index_entry lsda_entry;
                memcpy (&lsda_entry, lsda_entry_offset, sizeof (struct unwind_info_section_header_lsda_index_entry));
                printf ("    LSDA [%d] functionOffset %d (%d) (file address 0x%" PRIx64 "), lsdaOffset %d (file address 0x%" PRIx64 ")\n", 
                        lsda_count, lsda_entry.functionOffset, 
                        lsda_entry.functionOffset - index_entry.functionOffset, 
                        lsda_entry.functionOffset + baton.text_segment_vmaddr,
                        lsda_entry.lsdaOffset, lsda_entry.lsdaOffset + baton.text_segment_vmaddr);
                lsda_count++;
                lsda_entry_offset += sizeof (struct unwind_info_section_header_lsda_index_entry);
            }

            printf ("\n");

            baton.first_level_index_entry = index_entry;
            print_second_level_index (baton);
        }

        printf ("\n");

        cur_idx++;
        offset += sizeof (struct unwind_info_section_header_index_entry);
    }
}

int main (int argc, char **argv)
{
    struct stat st;
    char *file = argv[0];
    if (argc > 1)
        file = argv[1];
    int fd = open (file, O_RDONLY);
    if (fd == -1)
    {
        printf ("Failed to open '%s'\n", file);
        exit (1);
    }
    fstat (fd, &st);
    uint8_t *file_mem = (uint8_t*) mmap (0, st.st_size, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0);
    if (file_mem == MAP_FAILED)
    {
        printf ("Failed to mmap() '%s'\n", file);
    }

    FILE *f = fopen ("a.out", "r");

    struct baton baton;
    baton.mach_header_start = file_mem;
    scan_macho_load_commands (&baton);

    if (baton.compact_unwind_start == NULL)
    {
        printf ("could not find __TEXT,__unwind_info section\n");
        exit (1);
    }


    struct unwind_info_section_header header;
    memcpy (&header, baton.compact_unwind_start, sizeof (struct unwind_info_section_header));
    printf ("Header:\n");
    printf ("  version %u\n", header.version);
    printf ("  commonEncodingsArraySectionOffset is %d\n", header.commonEncodingsArraySectionOffset);
    printf ("  commonEncodingsArrayCount is %d\n", header.commonEncodingsArrayCount);
    printf ("  personalityArraySectionOffset is %d\n", header.personalityArraySectionOffset);
    printf ("  personalityArrayCount is %d\n", header.personalityArrayCount);
    printf ("  indexSectionOffset is %d\n", header.indexSectionOffset);
    printf ("  indexCount is %d\n", header.indexCount);

    uint8_t *common_encodings = baton.compact_unwind_start + header.commonEncodingsArraySectionOffset;
    uint32_t encoding_idx = 0;
    while (encoding_idx < header.commonEncodingsArrayCount)
    {
        uint32_t encoding = *((uint32_t*) common_encodings);
        printf ("    Common Encoding [%d]: 0x%x", encoding_idx, encoding);
        print_encoding (baton, encoding);
        printf ("\n");
        common_encodings += sizeof (uint32_t);
        encoding_idx++;
    }

    uint8_t *pers_arr = baton.compact_unwind_start + header.personalityArraySectionOffset;
    uint32_t pers_idx = 0;
    while (pers_idx < header.personalityArrayCount)
    {
        int32_t pers_delta = *((int32_t*) (baton.compact_unwind_start + header.personalityArraySectionOffset + (pers_idx * sizeof (uint32_t))));
        printf ("    Personality [%d]: offset to personality function address ptr %d (file address 0x%" PRIx64 ")\n", pers_idx, pers_delta, baton.text_segment_vmaddr + pers_delta);
        pers_idx++;
        pers_arr += sizeof (uint32_t);
    }

    printf ("\n");

    baton.unwind_header = header;

    print_index_sections (baton);


    return 0;
}
