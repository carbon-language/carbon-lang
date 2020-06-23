# Example: Incremental migration of APIs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [1. Start with a library in C++](#1-start-with-a-library-in-c)
- [2. Migrate the API to Carbon](#2-migrate-the-api-to-carbon)
- [3. Migrate callers to the Carbon API](#3-migrate-callers-to-the-carbon-api)
- [4. Clean up migration support](#4-clean-up-migration-support)
- [5. Migration complete](#5-migration-complete)

<!-- tocstop -->

In this example, we migrate an API to Carbon. The API is called by and also
calls C++ code, so bidirectional interoperability is used.

### 1. Start with a library in C++

C++ libraries may already be in use from Carbon; here, we assume a Carbon
TestServer and C++ `main.cc` both invoke APIs from the C++ `start_server.h`.

C++ `start_server.h`:

```cc
#include "net/util/ports.h"

namespace project {

void StartServerOnPort(int port) {
  ...
}

void StartServer() {
  StartServerOnPort(net_util::PickUnusedPortOrDie());
}

}  // namespace project
```

C++ `main.cc`:

```cc
#include "project/start_server.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon TestServer library:

```carbon
package Project library TestServer;

import Cpp "project/start_server.h";

fn TestStartup() {
  ...
  Cpp.project.StartServer();
  ...
}
```

### 2. Migrate the API to Carbon

This change must modify both the C++ and Carbon implementations in one change.
However, callers should not need to be modified, keeping this change local. By
including the generated

An extern may optionally be added to leave a stub call for C++ users, as in this
example. Carbon imports of C++ will also be able to use this, essentially
resulting in Carbon calling through C++ to call the migrated Carbon code. If all
C++ callers are being migrated to Carbon atomically in this change, it is not
necessary.

In this example, `main.cc` doesn't need to change yet because it transparently
uses the `$extern` provided by the Carbon StartServer library.

C++ `start_server.h`:

```cc
#include "net/util/ports.h"

// Include the Carbon file to make the externed StartServer symbol available.
#include "project/startserver.carbon.h"

namespace project {

void StartServerOnPort(int port) {
  ...
}

// The StartServer() call that was here is removed.

}  // namespace project
```

Carbon StartServer library:

```carbon
package Project library StartServer;

import Cpp "net/util/ports.h"
import Cpp "project/start_server.h";

// This replaces the alias.
$extern("Cpp", namespace="project") fn StartServer() {
  Cpp.project.StartServerOnPort(Cpp.net_util.PickUnusedPortOrDie());
}
```

### 3. Migrate callers to the Carbon API

Both C++ and Carbon callers should be migrated to the Carbon API now.

C++ `main.cc`:

```cc
// This replaces the include of the C++ startserver.h.
#include "project/startserver.carbon.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon TestServer library:

```carbon
package Project library TestServer;

// This replaces the import of the C++ start_server.h.
import Project library StartServer;

fn TestStartup() {
  ...
  // This replaces the call into C++.
  Project.StartServer.StartServer();
  ...
}
```

### 4. Clean up migration support

The `#include` added to `start_server.h` during step #2 is not necessary after
C++ callers are migrated to call into the Carbon implementation, and can be
removed.

### 5. Migration complete

This is repeating the above example code to summarize what the final migrated
code looks like.

C++ `start_server.h`:

```cc
#include "net/util/ports.h"

namespace project {

void StartServerOnPort(int port) {
  ...
}

}  // namespace project
```

C++ `main.cc`:

```cc
#include "project/startserver.carbon.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon StartServer library:

```carbon
package Project library StartServer;

import Cpp "net/util/ports.h"
import Cpp "project/start_server.h";

$extern("Cpp", namespace="project") fn StartServer() {
  Cpp.project.StartServerOnPort(Cpp.net_util.PickUnusedPortOrDie());
}
```

Carbon TestServer library:

```carbon
package Project library TestServer;

import Project library StartServer;

fn TestStartup() {
  ...
  Project.StartServer.StartServer();
  ...
}
```
