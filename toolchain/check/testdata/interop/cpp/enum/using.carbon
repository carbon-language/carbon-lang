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
  let col : Cpp.Color = Cpp.Color.Red;
  let colb : Cpp.Color = Cpp.Blue;
  let pet : Cpp.Zoo.Animal = Cpp.Zoo.Animal.Cat;
  let pet2 : Cpp.Zoo.Animal = Cpp.Dog;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- import_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Color: type = class_type @Color [concrete]
// CHECK:STDOUT:   %pattern_type.839: type = pattern_type %Color [concrete]
// CHECK:STDOUT:   %int_0.40a: %Color = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.a8e: %Color = int_value 1 [concrete]
// CHECK:STDOUT:   %Zoo: type = class_type @Zoo [concrete]
// CHECK:STDOUT:   %Animal: type = class_type @Animal [concrete]
// CHECK:STDOUT:   %pattern_type.d9d: type = pattern_type %Animal [concrete]
// CHECK:STDOUT:   %int_0.83f: %Animal = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.afa: %Animal = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Color = %Color.decl
// CHECK:STDOUT:     .Blue = %int_1.a8e
// CHECK:STDOUT:     .Zoo = %Zoo.decl
// CHECK:STDOUT:     .Dog = %int_1.afa
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Color.decl: type = class_decl @Color [concrete = constants.%Color] {} {}
// CHECK:STDOUT:   %int_0.40a: %Color = int_value 0 [concrete = constants.%int_0.40a]
// CHECK:STDOUT:   %int_1.a8e: %Color = int_value 1 [concrete = constants.%int_1.a8e]
// CHECK:STDOUT:   %Zoo.decl: type = class_decl @Zoo [concrete = constants.%Zoo] {} {}
// CHECK:STDOUT:   %Animal.decl: type = class_decl @Animal [concrete = constants.%Animal] {} {}
// CHECK:STDOUT:   %int_0.83f: %Animal = int_value 0 [concrete = constants.%int_0.83f]
// CHECK:STDOUT:   %int_1.afa: %Animal = int_value 1 [concrete = constants.%int_1.afa]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %col.patt: %pattern_type.839 = value_binding_pattern col [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Color.ref.loc18_28: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   %Red.ref: %Color = name_ref Red, imports.%int_0.40a [concrete = constants.%int_0.40a]
// CHECK:STDOUT:   %.loc18: type = splice_block %Color.ref.loc18_16 [concrete = constants.%Color] {
// CHECK:STDOUT:     %Cpp.ref.loc18_13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Color.ref.loc18_16: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %col: %Color = value_binding col, %Red.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %colb.patt: %pattern_type.839 = value_binding_pattern colb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Blue.ref: %Color = name_ref Blue, imports.%int_1.a8e [concrete = constants.%int_1.a8e]
// CHECK:STDOUT:   %.loc19: type = splice_block %Color.ref.loc19 [concrete = constants.%Color] {
// CHECK:STDOUT:     %Cpp.ref.loc19_14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Color.ref.loc19: type = name_ref Color, imports.%Color.decl [concrete = constants.%Color]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %colb: %Color = value_binding colb, %Blue.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pet.patt: %pattern_type.d9d = value_binding_pattern pet [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc20_30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Zoo.ref.loc20_33: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:   %Animal.ref.loc20_37: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   %Cat.ref: %Animal = name_ref Cat, imports.%int_0.83f [concrete = constants.%int_0.83f]
// CHECK:STDOUT:   %.loc20: type = splice_block %Animal.ref.loc20_20 [concrete = constants.%Animal] {
// CHECK:STDOUT:     %Cpp.ref.loc20_13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Zoo.ref.loc20_16: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:     %Animal.ref.loc20_20: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pet: %Animal = value_binding pet, %Cat.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pet2.patt: %pattern_type.d9d = value_binding_pattern pet2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc21_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dog.ref: %Animal = name_ref Dog, imports.%int_1.afa [concrete = constants.%int_1.afa]
// CHECK:STDOUT:   %.loc21: type = splice_block %Animal.ref.loc21 [concrete = constants.%Animal] {
// CHECK:STDOUT:     %Cpp.ref.loc21_14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Zoo.ref.loc21: type = name_ref Zoo, imports.%Zoo.decl [concrete = constants.%Zoo]
// CHECK:STDOUT:     %Animal.ref.loc21: type = name_ref Animal, imports.%Animal.decl [concrete = constants.%Animal]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pet2: %Animal = value_binding pet2, %Dog.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
