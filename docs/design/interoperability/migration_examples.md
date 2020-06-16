# Migration examples

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Large-scale migrations need to be done piecemeal, so the goal of migration
examples is to break down example migrations into small steps which can be
performed independently, without breaking surrounding code.

## Incremental migration of APIs

In this example, we migrate an API to Carbon. The API is called by and also
calls C++ code, so bidirectional interoperability is used.

### 1. Start with a library in C++

C++ libraries may already be in use from Carbon; here, we assume a Carbon
TestServer and C++ `main.cc` both invoke APIs from the C++ `start_server.h`.

C++ `start_server.h`:

```
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

```
#include "project/start_server.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon TestServer library:

```
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

An extern may optionally be added, to leave a stub call for C++ users (as in
this example). Carbon imports of C++ will also be able to use this, essentially
resulting in Carbon calling through C++ to call the migrated Carbon code. If all
C++ callers are being migrated to Carbon atomically in this change, it is not
necessary.

In this example, `main.cc` doesn't need to change yet because it transparently
uses the `$extern` provided by the Carbon StartServer library.

C++ `start_server.h`:

```
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

```
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

```
// This replaces the include of the C++ startserver.h.
#include "project/startserver.carbon.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon TestServer library:

```
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

```
#include "net/util/ports.h"

namespace project {

void StartServerOnPort(int port) {
  ...
}

}  // namespace project
```

C++ `main.cc`:

```
#include "project/startserver.carbon.h"

int main(int argc, char** argv) {
  ...
  project::StartServer();
  ...
}
```

Carbon StartServer library:

```
package Project library StartServer;

import Cpp "net/util/ports.h"
import Cpp "project/start_server.h";

$extern("Cpp", namespace="project") fn StartServer() {
  Cpp.project.StartServerOnPort(Cpp.net_util.PickUnusedPortOrDie());
}
```

Carbon TestServer library:

```
package Project library TestServer;

import Project library StartServer;

fn TestStartup() {
  ...
  Project.StartServer.StartServer();
  ...
}
```

## Framework API

### 1. Start with a library in C++

C++ `framework.h`:

```
class FrameworkBase {
  public:
    virtual void Run() = 0;
};

void Register(FrameworkBase* api) {
  ... registration logic ...
}
```

C++ `user.h`:

```
#include "framework.h"

class UserImpl : public FrameworkBase {
 public:
  void Run() override { ... }
};
```

### 2. Move registration to Carbon

Carbon Framework library:

```
package Framework;

// Provide an interface for Carbon users.
interface FrameworkInterface {
  fn Run();
}

fn Register(FrameworkInterface* api) {
  ... registration logic ...
}

// Provide a bridge structure for C++ users.
struct FrameworkBridge {
  impl FrameworkApi {
        fn Run() { impl_.Run(); }
  }
  private Cpp.FrameworkBase impl_;
}

// Replace the old register method with this bridge call.
$extern("Cpp", name="Register")
fn RegisterForCpp(Cpp.FrameworkBase* api) {
  Register(FrameworkBridge(api));
}
```

C++ `framework.h`

```
// Include the generated Carbon header for Register.
#include "framework.carbon.h"

class FrameworkBase {
  public:
    virtual void Run() = 0;
};

// Remove the Register() call here, because Carbon now provides it.
```

### 3. Migrate users to Carbon

At this point users should be able to migrate their implementations to Carbon
cleanly, with no wrapper code.

### 4. Clean up migration support

The RegisterForCpp method and C++ libraries may be removed after all users have
been migrated to Carbon.

### Caveat

Note that this approach for framework APIs uses double dynamic dispatch for C++
users (once for FrameworkInterface, once for FrameworkBase). It's likely the
added dynamic call cost is going to be negligible for most use-cases, given the
presumption of one dynamic call being present in the original C++ code. However,
some frameworks may need to examine their approaches, and consider migrating
performance-critical code differently.
