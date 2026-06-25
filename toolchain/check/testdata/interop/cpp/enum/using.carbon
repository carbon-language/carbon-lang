// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/using.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/using.carbon

// --- enum.h


enum class Color {Red, Blue, Green};

using enum Color;

class Zoo {
public:
  enum class Animal { Cat, Dog, Horse };
};

using enum Zoo::Animal;

// --- import_enum.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_enum.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./enum.h:5:7: warning: using enum declaration is a C++20 extension [CppInteropParseWarning]
// CHECK:STDERR:     5 | using enum Color;
// CHECK:STDERR:       |       ^
// CHECK:STDERR:
// CHECK:STDERR: import_enum.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./enum.h:12:7: warning: using enum declaration is a C++20 extension [CppInteropParseWarning]
// CHECK:STDERR:    12 | using enum Zoo::Animal;
// CHECK:STDERR:       |       ^
// CHECK:STDERR:
import Cpp library "enum.h";

//@dump-sem-ir-begin
fn F() {
  let unused col : Cpp.Color = Cpp.Color.Red;
  let unused colb : Cpp.Color = Cpp.Blue;
  let unused pet : Cpp.Zoo.Animal = Cpp.Zoo.Animal.Cat;
  let unused pet2 : Cpp.Zoo.Animal = Cpp.Dog;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- import_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Color: type = class_type @Color [concrete]
// CHECK:STDOUT:   %pattern_type.732: type = pattern_type %Color [concrete]
// CHECK:STDOUT:   %col.patt: %pattern_type.732 = value_binding_pattern col [concrete]
// CHECK:STDOUT:   %int_0.59e: %Color = int_value 0 [concrete]
// CHECK:STDOUT:   %colb.patt: %pattern_type.732 = value_binding_pattern colb [concrete]
// CHECK:STDOUT:   %int_1.b76: %Color = int_value 1 [concrete]
// CHECK:STDOUT:   %Zoo: type = class_type @Zoo [concrete]
// CHECK:STDOUT:   %Animal: type = class_type @Animal [concrete]
// CHECK:STDOUT:   %pattern_type.8cd: type = pattern_type %Animal [concrete]
// CHECK:STDOUT:   %pet.patt: %pattern_type.8cd = value_binding_pattern pet [concrete]
// CHECK:STDOUT:   %int_0.648: %Animal = int_value 0 [concrete]
// CHECK:STDOUT:   %pet2.patt: %pattern_type.8cd = value_binding_pattern pet2 [concrete]
// CHECK:STDOUT:   %int_1.e7b: %Animal = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Color = %Color.decl
// CHECK:STDOUT:     .Blue = %int_1.b76
// CHECK:STDOUT:     .Zoo = %Zoo.decl
// CHECK:STDOUT:     .Dog = %int_1.e7b
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Color.decl: type = class_decl @Color [concrete = constants.%Color] {} {}
// CHECK:STDOUT:   %int_0.59e: %Color = int_value 0 [concrete = constants.%int_0.59e]
// CHECK:STDOUT:   %int_1.b76: %Color = int_value 1 [concrete = constants.%int_1.b76]
// CHECK:STDOUT:   %Zoo.decl: type = class_decl @Zoo [concrete = constants.%Zoo] {} {}
// CHECK:STDOUT:   %Animal.decl: type = class_decl @Animal [concrete = constants.%Animal] {} {}
// CHECK:STDOUT:   %int_0.648: %Animal = int_value 0 [concrete = constants.%int_0.648]
// CHECK:STDOUT:   %int_1.e7b: %Animal = int_value 1 [concrete = constants.%int_1.e7b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %col.patt: %pattern_type.732 = value_binding_pattern col [concrete = constants.%col.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18_32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Color.ref.loc18_35: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   %Red.ref: %Color = name_ref Red, imports.%int_0.59e [concrete = constants.%int_0.59e]
// CHECK:STDOUT:   %.loc18: type = splice_block %Color.ref.loc18_23 [concrete = constants.%Color] {
// CHECK:STDOUT:     %Cpp.ref.loc18_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Color.ref.loc18_23: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %col: %Color = value_binding col, %Red.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %colb.patt: %pattern_type.732 = value_binding_pattern colb [concrete = constants.%colb.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Blue.ref: %Color = name_ref Blue, imports.%int_1.b76 [concrete = constants.%int_1.b76]
// CHECK:STDOUT:   %.loc19: type = splice_block %Color.ref.loc19 [concrete = constants.%Color] {
// CHECK:STDOUT:     %Cpp.ref.loc19_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Color.ref.loc19: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %colb: %Color = value_binding colb, %Blue.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pet.patt: %pattern_type.8cd = value_binding_pattern pet [concrete = constants.%pet.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc20_37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Zoo.ref.loc20_40: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:   %Animal.ref.loc20_44: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   %Cat.ref: %Animal = name_ref Cat, imports.%int_0.648 [concrete = constants.%int_0.648]
// CHECK:STDOUT:   %.loc20: type = splice_block %Animal.ref.loc20_27 [concrete = constants.%Animal] {
// CHECK:STDOUT:     %Cpp.ref.loc20_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Zoo.ref.loc20_23: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:     %Animal.ref.loc20_27: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pet: %Animal = value_binding pet, %Cat.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pet2.patt: %pattern_type.8cd = value_binding_pattern pet2 [concrete = constants.%pet2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc21_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dog.ref: %Animal = name_ref Dog, imports.%int_1.e7b [concrete = constants.%int_1.e7b]
// CHECK:STDOUT:   %.loc21: type = splice_block %Animal.ref.loc21 [concrete = constants.%Animal] {
// CHECK:STDOUT:     %Cpp.ref.loc21_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Zoo.ref.loc21: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:     %Animal.ref.loc21: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pet2: %Animal = value_binding pet2, %Dog.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
