//===- README.txt - Notes for improving CellSPU-specific code gen ---------===//

This code was contributed by a team from the Computer Systems Research
Department in The Aerospace Corporation:

- Scott Michel (head bottle washer and much of the non-floating point
  instructions)
- Mark Thomas (floating point instructions)
- Michael AuYeung (intrinsics)
- Chandler Carruth (LLVM expertise)
- Nehal Desai (debugging, RoadRunner SPU expertise)

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR
OTHERWISE.  IN NO EVENT SHALL THE AEROSPACE CORPORATION BE LIABLE FOR DAMAGES
OF ANY KIND OR NATURE WHETHER BASED IN CONTRACT, TORT, OR OTHERWISE ARISING
OUT OF OR IN CONNECTION WITH THE USE OF THE SOFTWARE INCLUDING, WITHOUT
LIMITATION, DAMAGES RESULTING FROM LOST OR CONTAMINATED DATA, LOST PROFITS OR
REVENUE, COMPUTER MALFUNCTION, OR FOR ANY SPECIAL, INCIDENTAL, CONSEQUENTIAL,
OR PUNITIVE  DAMAGES, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES OR
SUCH DAMAGES ARE FORESEEABLE.

---------------------------------------------------------------------------
--WARNING--:
--WARNING--: The CellSPU work is work-in-progress and "alpha" quality code.
--WARNING--:

If you are brave enough to try this code or help to hack on it, be sure
to add 'spu' to configure's --enable-targets option, e.g.:

        ./configure <your_configure_flags_here> \
           --enable-targets=x86,x86_64,powerpc,spu

---------------------------------------------------------------------------

TODO:
* Create a machine pass for performing dual-pipeline scheduling specifically
  for CellSPU, handle inserting branch prediction instructions.

* i32 instructions:

  * i32 division (work-in-progress)

* i64 support (see i64operations.c test harness):

  * shifts and comparison operators: done
  * sign and zero extension: done
  * addition: done
  * subtraction: needed
  * multiplication: work-in-progress

* i128 support:

  * zero extension: done
  * sign extension: needed
  * arithmetic operators (add, sub, mul, div): needed

* Double floating point support

  This was started. "What's missing?" to be filled in.

* Intrinsics

  Lots of progress. "What's missing/incomplete?" to be filled in.

===-------------------------------------------------------------------------===
