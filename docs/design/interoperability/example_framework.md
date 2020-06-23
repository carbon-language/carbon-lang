# Example: Framework API migration

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [1. Start with a library in C++](#1-start-with-a-library-in-c)
- [2. Move registration to Carbon](#2-move-registration-to-carbon)
- [3. Migrate users to Carbon](#3-migrate-users-to-carbon)
- [4. Clean up migration support](#4-clean-up-migration-support)
- [Caveat](#caveat)

<!-- tocstop -->

In this example, we migrate a framework to Carbon while continuing to support
C++ users.

### 1. Start with a library in C++

C++ `framework.h`:

```cc
class FrameworkBase {
  public:
    virtual void Run() = 0;
};

void Register(FrameworkBase* api) {
  ... registration logic ...
}
```

C++ `user.h`:

```cc
#include "framework.h"

class UserImpl : public FrameworkBase {
 public:
  void Run() override { ... }
};
```

### 2. Move registration to Carbon

Carbon Framework library:

```carbon
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

```cc
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

The `RegisterForCpp` method and C++ libraries may be removed after all users
have been migrated to Carbon.

### Caveat

Note that this approach for framework APIs uses double dynamic dispatch for C++
users: once for `FrameworkInterface` and once for `FrameworkBase`. It's likely
the added dynamic call cost is going to be negligible for most use-cases, given
the presumption of one dynamic call being present in the original C++ code.
However, some frameworks may need to examine their approaches, and consider
migrating performance-critical code differently.
