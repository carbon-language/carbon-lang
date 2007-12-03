//===- README.txt - Notes for improving CellSPU-specific code gen ---------===//

This code was contributed by a team from the Computer Systems Research
Department in The Aerospace Corporation:

- Scott Michel (head bottle washer and much of the non-floating point
  instructions)
- Mark Thomas (floating point instructions)
- Michael AuYeung (intrinsics)
- Chandler Carruth (LLVM expertise)

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
--WARNING--: The CellSPU work is work-in-progress and "alpha" quality code.
---------------------------------------------------------------------------

TODO:
* Finish branch instructions, branch prediction

  These instructions were started, but only insofar as to get llvm-gcc-4.2's
  crtbegin.ll working (which doesn't.)

* Double floating point support

  This was started. "What's missing?" to be filled in.

* Intrinsics

  Lots of progress. "What's missing/incomplete?" to be filled in.

===-------------------------------------------------------------------------===
