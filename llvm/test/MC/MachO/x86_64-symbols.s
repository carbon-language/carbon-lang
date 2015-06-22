// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -t | FileCheck %s

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

        .section foo, bar
        .long L4 + 1
        .long L35 + 1
        .long L36 + 1
        .long L37 + 1
        .long L38 + 1

// CHECK: Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D0 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D1 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D2 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __const (0x2)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D3 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __static_const (0x3)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: (0)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D4 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D5 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal4 (0x5)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D6 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal8 (0x6)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D7 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __literal16 (0x7)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D8 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __constructor (0x8)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D9 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __destructor (0x9)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D12 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __data (0xA)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D13 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __static_data (0xB)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D16 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __dyld (0xC)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D17 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __mod_init_func (0xD)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D18 ({{.*}})
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
// CHECK-NEXT:     Name: D20 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __class (0x10)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D21 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __meta_class (0x11)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D22 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cat_cls_meth (0x12)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D23 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cat_inst_meth (0x13)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D24 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __protocol (0x14)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D25 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __string_object (0x15)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D26 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cls_meth (0x16)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D27 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __inst_meth (0x17)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D28 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cls_refs (0x18)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D29 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __message_refs (0x19)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D30 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __symbols (0x1A)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D31 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __category (0x1B)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D32 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __class_vars (0x1C)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D33 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __instance_vars (0x1D)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D34 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __module_info (0x1E)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: (0)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D35 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: (0)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D36 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: (0)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D37 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __cstring (0x4)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: (0)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __selector_strs (0x1F)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: D38 ({{.*}})
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __selector_strs (0x1F)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
