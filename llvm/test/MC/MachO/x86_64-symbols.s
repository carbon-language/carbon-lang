// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -s -t | FileCheck %s

        .text
L0:
D0:
        .section	__TEXT,__text,regular,pure_instructions
L1:
D1:
        .const
L2:
D2:
        .static_const
L3:
D3:
        .cstring
L4:
D4:
        .literal4
L5:
D5:
        .literal8
L6:
D6:
        .literal16
L7:
D7:
        .constructor
L8:
D8:
        .destructor
L9:
D9:
//        .symbol_stub
//L10:
//D10:
//        .picsymbol_stub
//L11:
//D11:
        .data
L12:
D12:
        .static_data
L13:
D13:
//        .non_lazy_symbol_pointer
//L14:
//D14:
//        .lazy_symbol_pointer
//L15:
//D15:
        .dyld
L16:
D16:
        .mod_init_func
L17:
D17:
        .mod_term_func
L18:
D18:
        .const_data
L19:
D19:
        .objc_class
L20:
D20:
        .objc_meta_class
L21:
D21:
        .objc_cat_cls_meth
L22:
D22:
        .objc_cat_inst_meth
L23:
D23:
        .objc_protocol
L24:
D24:
        .objc_string_object
L25:
D25:
        .objc_cls_meth
L26:
D26:
        .objc_inst_meth
L27:
D27:
        .objc_cls_refs
L28:
D28:
        .objc_message_refs
L29:
D29:
        .objc_symbols
L30:
D30:
        .objc_category
L31:
D31:
        .objc_class_vars
L32:
D32:
        .objc_instance_vars
L33:
D33:
        .objc_module_info
L34:
D34:
        .objc_class_names
L35:
D35:
        .objc_meth_var_types
L36:
D36:
        .objc_meth_var_names
L37:
D37:
        .objc_selector_strs
L38:
D38:
//        .section __TEXT,__picsymbolstub4,symbol_stubs,none,16
//L39:
//D39:

// CHECK:      Sections [
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 0
// CHECK-NEXT:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x800000)
// CHECK-NEXT:       PureInstructions (0x800000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 1
// CHECK-NEXT:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 2
// CHECK-NEXT:     Name: __static_const (5F 5F 73 74 61 74 69 63 5F 63 6F 6E 73 74 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 3
// CHECK-NEXT:     Name: __cstring (5F 5F 63 73 74 72 69 6E 67 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: ExtReloc (0x2)
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 4
// CHECK-NEXT:     Name: __literal4 (5F 5F 6C 69 74 65 72 61 6C 34 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 2
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x3
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 5
// CHECK-NEXT:     Name: __literal8 (5F 5F 6C 69 74 65 72 61 6C 38 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 3
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: SomeInstructions (0x4)
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 6
// CHECK-NEXT:     Name: __literal16 (5F 5F 6C 69 74 65 72 61 6C 31 36 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 4
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0xE
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 7
// CHECK-NEXT:     Name: __constructor (5F 5F 63 6F 6E 73 74 72 75 63 74 6F 72 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 8
// CHECK-NEXT:     Name: __destructor (5F 5F 64 65 73 74 72 75 63 74 6F 72 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 9
// CHECK-NEXT:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 10
// CHECK-NEXT:     Name: __static_data (5F 5F 73 74 61 74 69 63 5F 64 61 74 61 00 00 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 11
// CHECK-NEXT:     Name: __dyld (5F 5F 64 79 6C 64 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 12
// CHECK-NEXT:     Name: __mod_init_func (5F 5F 6D 6F 64 5F 69 6E 69 74 5F 66 75 6E 63 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 2
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x9
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 13
// CHECK-NEXT:     Name: __mod_term_func (5F 5F 6D 6F 64 5F 74 65 72 6D 5F 66 75 6E 63 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 2
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0xA
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 14
// CHECK-NEXT:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 15
// CHECK-NEXT:     Name: __class (5F 5F 63 6C 61 73 73 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 16
// CHECK-NEXT:     Name: __meta_class (5F 5F 6D 65 74 61 5F 63 6C 61 73 73 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 17
// CHECK-NEXT:     Name: __cat_cls_meth (5F 5F 63 61 74 5F 63 6C 73 5F 6D 65 74 68 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 18
// CHECK-NEXT:     Name: __cat_inst_meth (5F 5F 63 61 74 5F 69 6E 73 74 5F 6D 65 74 68 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 19
// CHECK-NEXT:     Name: __protocol (5F 5F 70 72 6F 74 6F 63 6F 6C 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 20
// CHECK-NEXT:     Name: __string_object (5F 5F 73 74 72 69 6E 67 5F 6F 62 6A 65 63 74 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 21
// CHECK-NEXT:     Name: __cls_meth (5F 5F 63 6C 73 5F 6D 65 74 68 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 22
// CHECK-NEXT:     Name: __inst_meth (5F 5F 69 6E 73 74 5F 6D 65 74 68 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 23
// CHECK-NEXT:     Name: __cls_refs (5F 5F 63 6C 73 5F 72 65 66 73 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 2
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x5
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 24
// CHECK-NEXT:     Name: __message_refs (5F 5F 6D 65 73 73 61 67 65 5F 72 65 66 73 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 2
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x5
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 25
// CHECK-NEXT:     Name: __symbols (5F 5F 73 79 6D 62 6F 6C 73 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 26
// CHECK-NEXT:     Name: __category (5F 5F 63 61 74 65 67 6F 72 79 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 27
// CHECK-NEXT:     Name: __class_vars (5F 5F 63 6C 61 73 73 5F 76 61 72 73 00 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 28
// CHECK-NEXT:     Name: __instance_vars (5F 5F 69 6E 73 74 61 6E 63 65 5F 76 61 72 73 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 29
// CHECK-NEXT:     Name: __module_info (5F 5F 6D 6F 64 75 6C 65 5F 69 6E 66 6F 00 00 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x100000)
// CHECK-NEXT:       NoDeadStrip (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 30
// CHECK-NEXT:     Name: __selector_strs (5F 5F 73 65 6C 65 63 74 6F 72 5F 73 74 72 73 00)
// CHECK-NEXT:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 2688
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: ExtReloc (0x2)
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D0 (139)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D1 (128)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D2 (113)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __const (0x2)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D3 (98)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __static_const (0x3)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: L4 (84)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D4 (87)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D5 (69)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal4 (0x5)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D6 (50)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal8 (0x6)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D7 (31)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal16 (0x7)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D8 (12)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __constructor (0x8)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D9 (1)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __destructor (0x9)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D12 (124)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __data (0xA)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D13 (109)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __static_data (0xB)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D16 (65)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __dyld (0xC)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D17 (46)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __mod_init_func (0xD)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D18 (27)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __mod_term_func (0xE)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D19 (8)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __const (0xF)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D20 (146)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __class (0x10)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D21 (135)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __meta_class (0x11)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D22 (120)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cat_cls_meth (0x12)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D23 (105)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cat_inst_meth (0x13)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D24 (94)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __protocol (0x14)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D25 (80)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __string_object (0x15)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D26 (61)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cls_meth (0x16)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D27 (42)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __inst_meth (0x17)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D28 (23)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cls_refs (0x18)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D29 (4)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __message_refs (0x19)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D30 (142)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __symbols (0x1A)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D31 (131)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __category (0x1B)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D32 (116)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __class_vars (0x1C)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D33 (101)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __instance_vars (0x1D)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D34 (90)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __module_info (0x1E)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: L35 (72)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D35 (76)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: L36 (53)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D36 (57)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: L37 (34)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D37 (38)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: L38 (15)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __selector_strs (0x1F)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D38 (19)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __selector_strs (0x1F)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
