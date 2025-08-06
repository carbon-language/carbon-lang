// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --expect-full-prelude --exclude-dump-file-prefix=min_prelude/

// Including this file indicates explicitly to file_test that the test wants to
// include the full production prelude. Note that this is quite slow relative to
// a more minimal prelude (e.g. ~1s vs 10ms), and makes debugging tests quite
// noisy as the prelude compiles many things.

// --- min_prelude/full.carbon

// A no-op package.
package MinPreludeFull;
