// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";
import Core library "prelude/types/string";

fn Check(condition: bool, test_name: Core.String) -> bool {
  Core.PrintStr("[ RUN      ] ");
  Core.PrintStr(test_name);
  Core.PrintStr("\n");
  if (condition) {
    Core.PrintStr("[       OK ] ");
    Core.PrintStr(test_name);
    Core.PrintStr("\n");
    return true;
  } else {
    Core.PrintStr("[  FAILED  ] ");
    Core.PrintStr(test_name);
    Core.PrintStr(" - expectation failed\n");
    return false;
  }
}

fn Run() -> i32 {
  var passed: bool = true;

  // Test 1: Identical strings
  var s1: Core.String = "hello";
  var s2: Core.String = "hello";
  if (not Check(s1 == s2, "StringEquality_IdenticalEqual")) { passed = false; }
  if (not Check(not (s1 != s2), "StringEquality_IdenticalNotEqualFalse")) { passed = false; }

  // Test 2: Different characters, same length
  var s3: Core.String = "world";
  if (not Check(not (s1 == s3), "StringEquality_DiffCharsEqualFalse")) { passed = false; }
  if (not Check(s1 != s3, "StringEquality_DiffCharsNotEqual")) { passed = false; }

  // Test 3: Different lengths (short-circuit length check)
  var s4: Core.String = "hello world";
  if (not Check(not (s1 == s4), "StringEquality_DiffLengthsEqualFalse")) { passed = false; }
  if (not Check(s1 != s4, "StringEquality_DiffLengthsNotEqual")) { passed = false; }

  // Test 4: Empty string comparison
  var empty1: Core.String = "";
  var empty2: Core.String = "";
  if (not Check(empty1 == empty2, "StringEquality_EmptyStringsEqual")) { passed = false; }
  if (not Check(not (empty1 == s1), "StringEquality_EmptyAndNonEmptyEqualFalse")) { passed = false; }
  if (not Check(empty1 != s1, "StringEquality_EmptyAndNonEmptyNotEqual")) { passed = false; }

  // Test 5: Reflexivity (pointer identity fast-path)
  if (not Check(s1 == s1, "StringEquality_Reflexivity")) { passed = false; }

  if (passed) {
    Core.PrintStr("[  PASSED  ] All String equality tests passed.\n");
    return 0;
  } else {
    Core.PrintStr("[  FAILED  ] One or more String equality tests failed.\n");
    return 1;
  }
}
