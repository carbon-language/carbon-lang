//===---------------------------------------------------------------------===// 

Vector instructions support.

ShuffleVector
ExtractElement
InsertElement

//===---------------------------------------------------------------------===// 

Add "OpaqueType" type.

//===---------------------------------------------------------------------===// 

Variable argument functions support, "VAArg" instruction and some intrinsics.
CLI instruction "arglist" can get argument list descriptor for current function
that can be passed to other functions.

 va_arg = Function "System.ArgIterator::GetNextArg()"
 va_list = Class "System.ArgIterator"
 va_start = Instruction "arglist"
 va_end = Do nothing

//===---------------------------------------------------------------------===// 

If possible get used types inside of "MSILWriter" class, do not pass trougth
"MSILModule" class.

	getAnalysis<FindUsedTypes>().getTypes()

//===---------------------------------------------------------------------===// 

"switch" instruction emulation with CLI "switch" instruction.

//===---------------------------------------------------------------------===// 

Write linker for external function, because function export need to know 
dynamic library where function located.

.method static hidebysig pinvokeimpl("msvcrt.dll" cdecl)
	void free(void*) preservesig {}

//===---------------------------------------------------------------------===// 

DLLImport'ed variables support with the help of win32 envoronment.

	"LoadLibrary" - to load dynamic library into address space.
	"GetProcAddress" - to recive variable address by name.



