# REQUIRES: x86
# XFAIL: win32

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "LIBSAMPLE_1.0 { \
# RUN:   global:             \
# RUN:      extern "C++" {   \
# RUN:         \"foo()\";    \
# RUN:         \"zed()\";    \
# RUN:   };                  \
# RUN: };                    \
# RUN: LIBSAMPLE_2.0 {       \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       \"bar()\";      \
# RUN:   };                  \
# RUN: }; " > %t.script
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj -V -dyn-symbols %t.so | FileCheck --check-prefix=DSO %s

# DSO:      DynamicSymbols [
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: @
# DSO-NEXT:      Value: 0x0
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Local
# DSO-NEXT:      Type: None
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: Undefined
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3barv@@LIBSAMPLE_2.0
# DSO-NEXT:      Value: 0x1001
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3foov@@LIBSAMPLE_1.0
# DSO-NEXT:      Value: 0x1000
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3zedv@@LIBSAMPLE_1.0
# DSO-NEXT:      Value: 0x1002
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global (0x1)
# DSO-NEXT:      Type: Function (0x2)
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text (0x6)
# DSO-NEXT:    }
# DSO-NEXT:  ]
# DSO-NEXT:  Version symbols {
# DSO-NEXT:    Section Name: .gnu.version
# DSO-NEXT:    Address: 0x228
# DSO-NEXT:    Offset: 0x228
# DSO-NEXT:    Link: 1
# DSO-NEXT:    Symbols [
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 0
# DSO-NEXT:        Name: @
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 3
# DSO-NEXT:        Name: _Z3barv@@LIBSAMPLE_2.0
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 2
# DSO-NEXT:        Name: _Z3foov@@LIBSAMPLE_1.0
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 2
# DSO-NEXT:        Name: _Z3zedv@@LIBSAMPLE_1.0
# DSO-NEXT:      }
# DSO-NEXT:    ]
# DSO-NEXT:  }

.text
.globl _Z3foov
.type _Z3foov,@function
_Z3foov:
retq

.globl _Z3barv
.type _Z3barv,@function
_Z3barv:
retq

.globl _Z3zedv
.type _Z3zedv,@function
_Z3zedv:
retq
