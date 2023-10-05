# file_test

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## BUILD

A typical BUILD target will look like:

```
load("rules.bzl", "file_test")

file_test(
    name = "my_file_test",
    srcs = ["my_file_test.cpp"],
    tests = glob(["testdata/**"]),
    deps = [
        ":my_lib",
        "//testing/file_test:file_test_base",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
```

## Implementation

A typical implementation will look like:

```
#include "my_library.h"

#include "llvm/ADT/StringExtras.h"
#include "testing/file_test/file_test_base.h"

namespace Carbon::Testing {
namespace {

class MyFileTest : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  // Called as part of individual test executions.
  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           const llvm::SmallVector<TestFile>& test_files,
           llvm::raw_pwrite_stream& stdout, llvm::raw_pwrite_stream& stderr)
      -> ErrorOr<bool> override {
    MyFunctionality(test_args, stdout, stderr);
  }

  // Provides arguments which are used in tests that don't provide ARGS.
  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }
};

}  // namespace

// Registers for the framework to construct the tests.
CARBON_FILE_TEST_FACTORY(MyFileTest);

}  // namespace Carbon::Testing
```

## Comment markers

Settings in files are provided in comments, similar to `FileCheck` syntax.
`bazel run :file_test -- --autoupdate` automatically constructs compatible
CHECK:STDOUT: and CHECK:STDERR: lines.

Supported comment markers are:

-   ```
    // AUTOUDPATE
    // NOAUTOUPDATE
    ```

    Controls whether the checks in the file will be autoupdated if --autoupdate
    is passed. Exactly one of these two markers must be present. If the file
    uses splits, AUTOUPDATE must currently be before any splits.

    When autoupdating, CHECKs will be inserted starting below AUTOUPDATE. When a
    CHECK has line information, autoupdate will try to insert the CHECK
    immediately next to the line it's associated with, with stderr CHECKs
    preceding the line and stdout CHECKs following the line. When that happens,
    any subsequent CHECK lines without line information, or that refer to lines
    appearing earlier, will immediately follow. As an exception, if no STDOUT
    check line refers to any line in the test, all STDOUT check lines are placed
    at the end of the file instead of immediately after AUTOUPDATE.

-   `// ARGS: <arguments>`

    Provides a space-separated list of arguments, which will be passed to
    RunWithFiles as test_args. These are intended for use by the command as
    arguments.

    Supported replacements within arguments are:

    -   `%s`

        Replaced with the list of files. Currently only allowed as a standalone
        argument, not a substring.

    -   `%t`

        Replaced with `${TEST_TMPDIR}/temp_file`.

    ARGS can be specified at most once. If not provided, the FileTestBase child
    is responsible for providing default arguments.

-   `// SET-CHECK-SUBSET`

    By default, all lines of output must have a CHECK match. Adding this as a
    option sets it so that non-matching lines are ignored. All provided
    CHECK:STDOUT: and CHECK:STDERR: lines must still have a match in output.

    SET-CHECK-SUBSET can be specified at most once.

-   `// --- <filename>`

    By default, all file content is provided to the test as a single file in
    test_files. Using this marker allows the file to be split into multiple
    files which will all be passed to test_files.

    Files are not created on disk; it's expected the child will create an
    InMemoryFilesystem if needed.

-   ```
    // CHECK:STDOUT: <output line>
    // CHECK:STDERR: <output line>
    ```

    These provide a match for output from the command. See `SET-CHECK-SUBSET`
    for how to change from full to subset matching of output.

    Output line matchers may contain `[[@LINE+offset]` and `{{regex}}` syntaxes,
    similar to `FileCheck`.
