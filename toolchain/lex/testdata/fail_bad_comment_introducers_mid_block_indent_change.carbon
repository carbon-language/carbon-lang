// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// We only diagnose a bad comment introducer on the first line of a block of
// comments, but we only consider comments with identical indent to be a block.
// This file tests that specific behavior.

//abc
//abc
//abc
  //abc
  //abc
  //abc
//abc
//abc
//abc

// We move the auto-update marker to the end to ensure that the added CHECKs
// don't interrupt the specific block structures above.
// AUTOUPDATE
