# file_test

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!--
{% raw %}
Hides `{{` from jekyll's liquid parsing. Note endraw at the bottom.
-->

## BUILD

A typical BUILD target will look like:

```starlark
load("rules.bzl", "file_test")

file_test(
    name = "my_file_test",
    srcs = ["my_file_test.cpp"],
    tests = glob(["testdata/**"]),
    deps = [
        ":my_lib",
        "//testing/file_test:file_test_base",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
```

## Implementation

A typical implementation will look like:

```cpp
#include "my_library.h"

#include "testing/file_test/file_test_base.h"

namespace Carbon::Testing {
namespace {

class MyFileTest : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  // Called as part of individual test executions.
  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           const llvm::SmallVector<TestFile>& test_files,
           FILE* input_stream, llvm::raw_pwrite_stream& output_stream,
           llvm::raw_pwrite_stream& error_stream)
      -> ErrorOr<RunResult> override {
    return MyFunctionality(test_args, input_stream, output_stream,
                           error_stream);
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

## Filename `fail_` prefixes

When a run fails, information about what pieces failed are returned on
`RunResult`. This affects whether a `fail_` prefix on the file is required,
including in combination with split-file tests (using the `// --- <filename>`
comment marker).

The main test file and any split-files must have a `fail_` prefix if and only if
they have an associated error. An exception is that the main test file may omit
`fail_` when it contains split-files that have a `fail_` prefix.

## Content replacement

Some keywords can be inserted for content:

-   ```
    [[@LSP:<method>:<extra content>]]
    [[@LSP-CALL:<method>:<extra content>]]
    [[@LSP-NOTIFY:<method>:<extra content>]]
    [[@LSP-REPLY:<id>:<extra content>]]
    ```

    Produces JSON for an LSP method call, notification, or reply. Each includes
    the `Content-Length` header. The `:<extra content>` is optional, and may be
    omitted.

    The difference between the replacements mirrors LSP calling conventions:

    -   `LSP` does a minimal JSON format, and is primarily intended for testing
        errors.
        -   `method`: Assigned from `<method>`.
    -   `LSP-CALL` maps to
        [Request Message](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#requestMessage).
        -   `method`: Assigned from `<method>`.
        -   `id`: Automatically incremented across calls.
        -   `params`: Optionally assigned from `<extra content>`.
    -   `LSP-NOTIFY` maps to
        [Notification Message](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#notificationMessage).
        -   `method`: Assigned from `<method>`.
        -   `params`: Optionally assigned from `<extra content>`.
    -   `LSP-REPLY` maps to
        [Response Message](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#responseMessage).
        -   `id`: Assigned from `<id>`.
        -   `result`: Optionally assigned from `<extra content>`.

    Additional substitutions are made to `<extra contents>` in the following
    cases:

    -   ```
        [[@LSP-NOTIFY:textDocument/didOpen:"textDocument": {
            "uri": "file:/<filename>",
            "text": "FROM_FILE_SPLIT"
        }]]
        ```

        The keyword `FROM_FILE_SPLIT` is substituted with the content of the
        file split `<filename>`. All other properties are unchanged.

-   ```
    [[@TEST_NAME]]
    ```

    Replaces with the test name, which is the filename with the extension and
    any `fail_` or `todo_` prefixes removed. For split files, this is based on
    the split filename.

The `[[@` string is reserved for future replacements, but `[[` is allowed in
content (comment markers don't allow `[[`).

## Comment markers

Settings in files are provided in comments, similar to `FileCheck` syntax.
`bazel run :file_test -- --autoupdate` automatically constructs compatible
CHECK:STDOUT: and CHECK:STDERR: lines.

Supported comment markers are:

-   ```
    // AUTOUDPATE
    // NOAUTOUPDATE
    ```

    Controls whether the checks in the file will be autoupdated if
    `--autoupdate` is passed. Exactly one of these markers must be present. If
    the file uses splits, the marker must currently be before any splits.

    When autoupdating, `CHECK`s will be inserted starting below `AUTOUPDATE`.
    When a `CHECK` has line information, autoupdate will try to insert the
    `CHECK` immediately next to the line it's associated with, with stderr
    `CHECK`s preceding the line and stdout `CHECK`s following the line. When
    that happens, any subsequent `CHECK` lines without line information, or that
    refer to lines appearing earlier, will immediately follow. As an exception,
    if no `STDOUT` check line refers to any line in the test, all `STDOUT` check
    lines are placed at the end of the file instead of immediately after
    `AUTOUPDATE`.

    When using split files, if the last split file is named
    `// --- AUTOUPDATE-SPLIT`, all `CHECK`s will be added there; no line
    associations occur.

-   ```
    // ARGS: <arguments>
    ```

    Provides a space-separated list of arguments, which will be passed to
    RunWithFiles as test_args. These are intended for use by the command as
    arguments.

    Supported replacements within arguments are:

    -   `%s`

        Replaced with the list of files. Currently only allowed as a standalone
        argument, not a substring.

    -   `%t`

        Replaced with `${TEST_TMPDIR}/temp_file`.

    -   `%{identifier}`

        Replaces some implementation-specific identifier with a value. (Mappings
        provided by way of an optional `MyFileTest::GetArgReplacements`)

    `ARGS` can be specified at most once. If not provided, the `FileTestBase`
    child is responsible for providing default arguments.

-   ```
    // EXTRA-ARGS: <arguments>
    ```

    Same as `ARGS`, including substitution behavior, but appends to the default
    argument list instead of replacing it.

    `EXTRA-ARGS` can be specified at most once, and a test cannot specify both
    `ARGS` and `EXTRA-ARGS`.

-   ```
    // INCLUDE-FILE: <path/from/repository/root>
    ```

    Includes the specified file in the test's virtual file system and adds the
    path to the file's splits.

    This can be used to provide a minimal `Core` package (and `prelude` library)
    in toolchain tests. See the
    [toolchain docs](/toolchain/docs/adding_features.md) for how.

-   ```
    // SET-CAPTURE-CONSOLE-OUTPUT
    ```

    By default, stderr and stdout are expected to be piped through provided
    streams. Adding this causes the test's own stderr and stdout to be captured
    and added as well.

    This should be avoided because we are partly ensuring that streams are an
    API, but is helpful when wrapping Clang, where stderr is used directly.

    `SET-CAPTURE-CONSOLE-OUTPUT` can be specified at most once.

-   ```
    // SET-CHECK-SUBSET
    ```

    By default, all lines of output must have a `CHECK` match. Adding this as a
    option sets it so that non-matching lines are ignored. All provided
    `CHECK:STDOUT:` and `CHECK:STDERR:` lines must still have a match in output.

    `SET-CHECK-SUBSET`can be specified at most once.

-   ```
    // --- <filename>
    ```

    By default, all file content is provided to the test as a single file in
    `test_files`. Using this marker allows the file to be split into multiple
    files which will all be passed to `test_files`.

    Files are not created on disk; instead, content is passed in through the
    `fs` passed to `Run`.

    If the filename is `STDIN`, it will be provided as `input_stream` instead of
    in `test_files`. Currently, autoupdate can place `CHECK` lines in the
    `STDIN` split; use `AUTOUPDATE-SPLIT` to avoid that (see `AUTOUPDATE` for
    information).

-   ```
    // CHECK:STDOUT: <output line>
    // CHECK:STDERR: <output line>
    ```

    These provide a match for output from the command. See `SET-CHECK-SUBSET`
    for how to change from full to subset matching of output.

    Output line matchers may contain `[[@LINE+offset]` and `{{regex}}` syntaxes,
    similar to `FileCheck`.

-   ```
    // TIP: <tip>
    ```

    Tips like this are added by autoupdate, for example providing commands to
    run the test directly. Tips have no impact on validation; the marker informs
    autoupdate that it can update or remove them as needed.

<!--
{% endraw %}
-->
