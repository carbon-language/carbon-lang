## In brief:
* Use *clang-format* to resolve all layout questions.
* Where LLVM's C++ style guide is clear on usage, follow it.
* Otherwise, where a clear precedent exists in the project, follow it.
* Otherwise, where a good public C++ style guide is relevant and clear,
  follow it.  [Google's](https://google.github.io/styleguide/cppguide.html)
  is pretty good and comes with lots of justifications for its rules.
## In particular:
### Files
1. File names should use dashes, not underscores.  C++ sources have the
extension ".cc", not ".C" or ".cpp" or ".cxx".  Don't create needless
source directory hierarchies.
1. Header files should be idempotent.  Use the usual "#ifndef FORTRAN_header_H_",
"#define FORTRAN_header_H_", and "#endif  // FORTRAN_header_H_" technique.
1. #include every header defining an entity that your project header or source
file actually uses directly.  (Exception: when foo.cc starts, as it should,
with #include "foo.h", and foo.h includes bar.h in order to define the
interface to the module foo, you don't have to redundantly #include "bar.h"
in foo.cc.)
1. In the source file "foo.cc", put the #include of "foo.h" first.
Then #include other project headers in alphabetic order; then C++ standard
headers, also alphabetically; then C and system headers.
1. Don't include the standard iostream header.  If you need it for debugging,
remove the inclusion before committing.
### Naming
1. C++ names that correspond to STL names should look like those STL names
(e.g., *clear()* and *size()* member functions in a class that implements
a container).
1. Non-public data members should be named with leading miniscule (lower-case)
letters, internal camelCase capitalization, and a trailing underscore,
e.g. "DoubleEntryBookkeepingSystem myLedger_;".  POD structures with
only public data members shouldn't use trailing underscores, since they
don't have class functions in which data members need to be distinguishable.
1. Accessor member functions are named with the non-public data member's name,
less the trailing underscore.  Mutator member functions are named *set_...*
and should return "*this".  Don't define accessors or mutators needlessly.
1. Other class functions should be named with leading capital letters,
CamelCase, and no underscores, and, like all functions, should be based
on imperative verbs, e.g. *HaltAndCatchFire()*.
1. It is fine to use short names for local variables with limited scopes,
especially when you can declare them directly in a for()/while()/if()
condition.  Otherwise, prefer complete English words to abbreviations
when creating names.
### Commentary
1. Use // for all comments except for short notes within expressions.
1. When // follows code on a line, precede it with two spaces.
1. Comments should matter.  Assume that the reader knows current C++ at least as
well as you do and avoid distracting her by calling out usage of new
features in comments.
### Layout
Always run *clang-format* before committing code.  Other developers should
be able to run "git pull", then *clang-format*, and see only their own
changes.

Here's what you can expect to see *clang-format* do:
1. Indent with two spaces.
1. Don't indent public:, protected:, and private:
accessibility labels.
1. Never use more than 80 characters per source line.
1. Don't use tabs.
1. Don't indent the bodies of namespaces, even when nested.
1. Function result types go on the same line as the function and argument
names.

Don't try to make columns of variable names or comments
align vertically -- they are maintenance problems.

Always wrap the bodies of if(), else, while(), for(), do, &c.
with braces, even when the body is a single statement or empty.  The
opening { goes on
the end of the line, not on the next line.  Functions also put the opening
{ after the formal arguments or new-style result type, not on the next
line.  Use {} for empty inline constructors and destructors in classes.

Don't waste space on the screen with needless blank lines or elaborate block
commentary (lines of dashes, boxes of asterisks, &c.).  Write code so as to be
easily read and understood with a minimum of scrolling.
### C++ language
Use *C++17*, unless some compiler to which we must be portable lacks a feature
you are considering.
1. Never throw or catch exceptions.
1. Never use run-time type information or dynamic_cast<>.
1. Never declare static data that executes a constructor.
Use {braced initializers} in all circumstances where they work, including
default data member initialization.  They inhibit implicit truncation.
Don't use "= expr" initialization just to effect implicit truncation;
prefer an explicit static_cast<>.
1. Avoid unsigned types apart from size_t, which must be used with care.
When *int* just obviously works, just use *int*.  When you need something
bigger than *int*, use std::int64_t rather than *long* or long long.
1. Use namespaces to avoid conflicts with client code.  Use one top-level
project namespace.  Don't introduce needless nested namespaces within a
project when names don't conflict or better solutions exist.  Never use
"using namespace ...;", especially not "using namespace std;".  Access
STL entities with names like std::unique_ptr<>, without a leading "::".
1. Prefer static functions to functions in anonymous namespaces in source files.
1. Use *auto* judiciously.  When the type of a local variable is known,
monomorphic, and easy to type, be explicit rather than using *auto*.
1. Use move semantics and smart pointers to make dynamic memory ownership
clear.  Consider reworking any code that uses malloc() or a (non-placement)
operator new.
1. Use references for const arguments; prefer const references to values for
all but small types that are trivially copyable (e.g., *int*).  Use non-const
pointers for output arguments.  Put output arguments last (pace the standard
C library conventions for memcpy() & al.).
1. Prefer *typename* to *class* in template argument declarations.
1. Prefer enum class to plain enum wherever enum class will work.
1. Use constexpr and const generously.
1. When a switch() statement's labels do not cover all possible case values
explicitly, it should contains either a "default:;" at its end or a
default: label that obviously crashes.
#### Classes
1. Define only POD structures with struct.
1. Don't use "this->" in (non-static) member functions.
1. Define accessor and mutator member functions (implicitly) inline in the
class, after constructors and assignments.  Don't needlessly define
(implicit) inline member functions in classes unless they really solve a
performance problem.
1. Try to make class definitions in headers concise specifications of
interfaces, at least to the extent that C++ allows.
1. When copy constructors and copy assignment are not necessary,
and move constructors/assignment is present, don't declare them and they
will be implicitly deleted.  When neither copy nor move constructors
or assignments should exist for a class, explicitly delete all of them.
1. Make single-argument constructors (other than copy and move constructors)
explicit unless you really want to define an implicit conversion.
#### Overall design preferences
Don't use dynamic solutions to solve problems that can be solved at
build time; don't solve build time problems by writing programs that
produce source code when macros and templates suffice; don't write macros
when templates suffice.  Templates are statically typed, checked by the
compiler, and are (or should be) visible to debuggers.
