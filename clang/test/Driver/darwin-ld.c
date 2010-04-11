// Check that ld gets arch_multiple.

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -arch i386 -arch x86_64 %s -### -o foo 2> %t.log
// RUN: grep '".*ld.*" .*"-arch_multiple" "-final_output" "foo"' %t.log

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -filelist FOO -static 2> %t.log
// RUN: grep '"-lcrt0.o" .*"-lgcc_static"' %t.log
// RUN: grep '"-lgcc"' %t.log | count 0
// RUN: %clang -ccc-host-triple i386-apple-darwin7 -### -filelist FOO 2> %t.log
// RUN: grep '"-lcrt1.o" .*"-lgcc" "-lSystem"' %t.log
// RUN: grep '"-lgcc_s"' %t.log | count 0
// RUN: %clang -ccc-host-triple i386-apple-darwin8 -### -filelist FOO 2> %t.log
// RUN: grep '"-lcrt1.o" .*"-lgcc_s.10.4" "-lgcc" "-lSystem"' %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -filelist FOO 2> %t.log
// RUN: grep '"-lcrt1.10.5.o" .*"-lgcc_s.10.5" "-lgcc" "-lSystem"' %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin10 -### -filelist FOO 2> %t.log
// RUN: grep '"-lcrt1.10.6.o" .*"-lSystem" "-lgcc"' %t.log
// RUN: grep '"-lgcc_s"' %t.log | count 0

// Make sure we run dsymutil on source input files.
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -g %s -o BAR 2> %t.log
// RUN: grep '".*dsymutil" "BAR"' %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -g -filelist FOO %s -o BAR 2> %t.log
// RUN: grep '".*dsymutil" "BAR"' %t.log

// Splatter test case. This is gross, but it works for now. For the
// driver, just getting coverage of the tool code and checking the
// output options is nearly good enough. The main thing we are
// protecting against here is unintended changes in the driver
// output. Intended changes should add more reasonable test cases, and
// just update this test to match the expected behavior.
//
// Note that at conception, this exactly matches gcc.

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -A ARG0 -F ARG1 -L ARG2 -Mach -T ARG4 -X -Z -all_load -allowable_client ARG8 -bind_at_load -compatibility_version ARG11 -current_version ARG12 -d -dead_strip -dylib_file ARG14 -dylinker -dylinker_install_name ARG16 -dynamic -dynamiclib -e ARG19 -exported_symbols_list ARG20 -fexceptions -flat_namespace -fnested-functions -fopenmp -force_cpusubtype_ALL -fpie -fprofile-arcs -headerpad_max_install_names -image_base ARG29 -init ARG30 -install_name ARG31 -m ARG33 -miphoneos-version-min=2.0 -mmacosx-version-min=10.3.2 -multi_module -multiply_defined ARG37 -multiply_defined_unused ARG38 -no_dead_strip_inits_and_terms -nodefaultlibs -nofixprebinding -nomultidefs -noprebind -noseglinkedit -nostartfiles -nostdlib -pagezero_size ARG54 -pg -prebind -prebind_all_twolevel_modules -preload -r -read_only_relocs ARG55 -s -sectalign ARG57_0 ARG57_1 ARG57_2 -sectcreate ARG58_0 ARG58_1 ARG58_2 -sectobjectsymbols ARG59_0 ARG59_1 -sectorder ARG60_0 ARG60_1 ARG60_2 -seg1addr ARG61 -seg_addr_table ARG62 -seg_addr_table_filename ARG63 -segaddr ARG64_0 ARG64_1 -segcreate ARG65_0 ARG65_1 ARG65_2 -seglinkedit -segprot ARG67_0 ARG67_1 ARG67_2 -segs_read_FOO -segs_read_only_addr ARG69 -segs_read_write_addr ARG70 -shared-libgcc -single_module -static -static-libgcc -sub_library ARG77 -sub_umbrella ARG78 -t -twolevel_namespace -twolevel_namespace_hints -u ARG82 -umbrella ARG83 -undefined ARG84 -unexported_symbols_list ARG85 -w -weak_reference_mismatches ARG87 -whatsloaded -whyload -y -filelist FOO -l FOO 2> %t.log
// RUN: grep '".*ld.*" "-static" "-dylib" "-dylib_compatibility_version" "ARG11" "-dylib_current_version" "ARG12" "-arch" "i386" "-dylib_install_name" "ARG31" "-all_load" "-allowable_client" "ARG8" "-bind_at_load" "-dead_strip" "-no_dead_strip_inits_and_terms" "-dylib_file" "ARG14" "-dynamic" "-exported_symbols_list" "ARG20" "-flat_namespace" "-headerpad_max_install_names" "-image_base" "ARG29" "-init" "ARG30" "-macosx_version_min" "10.3.2" "-iphoneos_version_min" "2.0" "-nomultidefs" "-multi_module" "-single_module" "-multiply_defined" "ARG37" "-multiply_defined_unused" "ARG38" "-pie" "-prebind" "-noprebind" "-nofixprebinding" "-prebind_all_twolevel_modules" "-read_only_relocs" "ARG55" "-sectcreate" "ARG58_0" "ARG58_1" "ARG58_2" "-sectorder" "ARG60_0" "ARG60_1" "ARG60_2" "-seg1addr" "ARG61" "-segprot" "ARG67_0" "ARG67_1" "ARG67_2" "-segaddr" "ARG64_0" "ARG64_1" "-segs_read_only_addr" "ARG69" "-segs_read_write_addr" "ARG70" "-seg_addr_table" "ARG62" "-seg_addr_table_filename" "ARG63" "-sub_library" "ARG77" "-sub_umbrella" "ARG78" "-twolevel_namespace" "-twolevel_namespace_hints" "-umbrella" "ARG83" "-undefined" "ARG84" "-unexported_symbols_list" "ARG85" "-weak_reference_mismatches" "ARG87" "-X" "-y" "-w" "-pagezero_size" "ARG54" "-segs_read_FOO" "-seglinkedit" "-noseglinkedit" "-sectalign" "ARG57_0" "ARG57_1" "ARG57_2" "-sectobjectsymbols" "ARG59_0" "ARG59_1" "-segcreate" "ARG65_0" "ARG65_1" "ARG65_2" "-whyload" "-whatsloaded" "-dylinker_install_name" "ARG16" "-dylinker" "-Mach" "-d" "-s" "-t" "-Z" "-u" "ARG82" "-undefined" "ARG84" "-A" "ARG0" "-e" "ARG19" "-m" "ARG33" "-r" "-o" "a.out" "-L" "ARG2" "-lgomp".* "-filelist" "FOO" "-lFOO" "-lgcov" "-allow_stack_execute" "-T" "ARG4" "-F" "ARG1"' %t.log

// Don't run dsymutil on a fat build of an executable.
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch i386 -arch x86_64 -g %s 2> %t.log
// RUN: grep dsymutil %t.log | count 0

// Check linker changes that came with new linkedit format.
// RUN: touch %t.o
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 %t.o 2> %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv6 -miphoneos-version-min=3.0 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_0 %s < %t.log

// LINK_IPHONE_3_0: ld"
// LINK_IPHONE_3_0-NOT: -lcrt1.3.1.o
// LINK_IPHONE_3_0: -lcrt1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: ld"
// LINK_IPHONE_3_0: -dylib
// LINK_IPHONE_3_0: -ldylib1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: ld"
// LINK_IPHONE_3_0: -lbundle1.o
// LINK_IPHONE_3_0: -lSystem

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 %t.o 2> %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -arch armv7 -miphoneos-version-min=3.1 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_1 %s < %t.log

// LINK_IPHONE_3_1: ld"
// LINK_IPHONE_3_1-NOT: -lcrt1.o
// LINK_IPHONE_3_1: -lcrt1.3.1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: ld"
// LINK_IPHONE_3_1: -dylib
// LINK_IPHONE_3_1-NOT: -ldylib1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: ld"
// LINK_IPHONE_3_1-NOT: -lbundle1.o
// LINK_IPHONE_3_1: -lSystem
