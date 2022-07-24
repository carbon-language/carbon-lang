# Local installer

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This installs carbon-explorer locally for experimental use on Mac/Linux. It's
not recommended for most users. Bear in mind that Carbon Language is
**experimental** and still has no compiler. See
[our project status](/README.md#project-status) for more information.

To install to /usr, run:

```shell
bazel run -c opt //installers/local:install
```

To use a custom install path, run:

```shell
bazel run -c opt //installers/local:install \
  --//installers/local:install_path=/my/path
```

To uninstall, run:

```shell
bazel run //installers/local:uninstall
```

The build mode is important for installs, as debug versions may be installed. It
is not relevant for uninstalls.

If a custom install path is passed to :install, the same should be passed to
:uninstall.

Note this is just scripting paths: it's not intended to be a full-fledged
install.
