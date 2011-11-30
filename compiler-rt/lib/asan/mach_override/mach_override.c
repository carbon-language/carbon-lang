/*******************************************************************************
	mach_override.c
		Copyright (c) 2003-2009 Jonathan 'Wolf' Rentzsch: <http://rentzsch.com>
		Some rights reserved: <http://opensource.org/licenses/mit-license.php>

	***************************************************************************/

#include "mach_override.h"

#include <mach-o/dyld.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <sys/mman.h>

#include <CoreServices/CoreServices.h>

//#define DEBUG_DISASM 1
#undef DEBUG_DISASM

/**************************
*	
*	Constants
*	
**************************/
#pragma mark	-
#pragma mark	(Constants)

#if defined(__ppc__) || defined(__POWERPC__)

long kIslandTemplate[] = {
	0x9001FFFC,	//	stw		r0,-4(SP)
	0x3C00DEAD,	//	lis		r0,0xDEAD
	0x6000BEEF,	//	ori		r0,r0,0xBEEF
	0x7C0903A6,	//	mtctr	r0
	0x8001FFFC,	//	lwz		r0,-4(SP)
	0x60000000,	//	nop		; optionally replaced
	0x4E800420 	//	bctr
};

#define kAddressHi			3
#define kAddressLo			5
#define kInstructionHi		10
#define kInstructionLo		11

#elif defined(__i386__) 

#define kOriginalInstructionsSize 16

char kIslandTemplate[] = {
	// kOriginalInstructionsSize nop instructions so that we 
	// should have enough space to host original instructions 
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
	// Now the real jump instruction
	0xE9, 0xEF, 0xBE, 0xAD, 0xDE
};

#define kInstructions	0
#define kJumpAddress    kInstructions + kOriginalInstructionsSize + 1
#elif defined(__x86_64__)

#define kOriginalInstructionsSize 32

#define kJumpAddress    kOriginalInstructionsSize + 6

char kIslandTemplate[] = {
	// kOriginalInstructionsSize nop instructions so that we 
	// should have enough space to host original instructions 
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 
	0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
	// Now the real jump instruction
	0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
};

#endif

#define	kAllocateHigh		1
#define	kAllocateNormal		0

/**************************
*	
*	Data Types
*	
**************************/
#pragma mark	-
#pragma mark	(Data Types)

typedef	struct	{
	char	instructions[sizeof(kIslandTemplate)];
	int		allocatedHigh;
}	BranchIsland;

/**************************
*	
*	Funky Protos
*	
**************************/
#pragma mark	-
#pragma mark	(Funky Protos)

	mach_error_t
allocateBranchIsland(
		BranchIsland	**island,
		int				allocateHigh,
		void *originalFunctionAddress);

	mach_error_t
freeBranchIsland(
		BranchIsland	*island );

#if defined(__ppc__) || defined(__POWERPC__)
	mach_error_t
setBranchIslandTarget(
		BranchIsland	*island,
		const void		*branchTo,
		long			instruction );
#endif 

#if defined(__i386__) || defined(__x86_64__)
mach_error_t
setBranchIslandTarget_i386(
						   BranchIsland	*island,
						   const void		*branchTo,
						   char*			instructions );
void 
atomic_mov64(
		uint64_t *targetAddress,
		uint64_t value );

	static Boolean 
eatKnownInstructions( 
	unsigned char	*code, 
	uint64_t		*newInstruction,
	int				*howManyEaten, 
	char			*originalInstructions );
#endif

/*******************************************************************************
*	
*	Interface
*	
*******************************************************************************/
#pragma mark	-
#pragma mark	(Interface)

#if defined(__x86_64__)
mach_error_t makeIslandExecutable(void *address) {
	mach_error_t err = err_none;
    vm_size_t pageSize;
    host_page_size( mach_host_self(), &pageSize );
    uint64_t page = (uint64_t)address & ~(uint64_t)(pageSize-1);
    int e = err_none;
    e |= mprotect((void *)page, pageSize, PROT_EXEC | PROT_READ | PROT_WRITE);
    e |= msync((void *)page, pageSize, MS_INVALIDATE );
    if (e) {
        err = err_cannot_override;
    }
    return err;
}
#endif

    mach_error_t
mach_override_ptr(
	void *originalFunctionAddress,
    const void *overrideFunctionAddress,
    void **originalFunctionReentryIsland )
{
	assert( originalFunctionAddress );
	assert( overrideFunctionAddress );

	long	*originalFunctionPtr = (long*) originalFunctionAddress;
#ifdef DEBUG_DISASM
  {
    fprintf(stderr, "Replacing function at %p\n", originalFunctionAddress);
    fprintf(stderr, "First 16 bytes of the function: ");
    unsigned char *orig = (unsigned char *)originalFunctionAddress;
    int i;
    for (i = 0; i < 16; i++) {
       fprintf(stderr, "%x ", (unsigned int) orig[i]);
    }
    fprintf(stderr, "\n");
    fprintf(stderr,
            "To disassemble, save the following function as disas.c"
            " and run:\n  gcc -c disas.c && gobjdump -d disas.o\n"
            "The first 16 bytes of the original function will start"
            " after four nop instructions.\n");
    fprintf(stderr, "\nvoid foo() {\n  asm volatile(\"nop;nop;nop;nop;\");\n");
    int j = 0;
    for (j = 0; j < 2; j++) {
      fprintf(stderr, "  asm volatile(\".byte ");
      for (i = 8 * j; i < 8 * (j+1) - 1; i++) {
        fprintf(stderr, "0x%x, ", (unsigned int) orig[i]);
      }
      fprintf(stderr, "0x%x;\");\n", (unsigned int) orig[8 * (j+1) - 1]);
    }
    fprintf(stderr, "}\n\n");
  }
#endif

	mach_error_t	err = err_none;
	
#if defined(__ppc__) || defined(__POWERPC__)
	//	Ensure first instruction isn't 'mfctr'.
	#define	kMFCTRMask			0xfc1fffff
	#define	kMFCTRInstruction	0x7c0903a6
	
	long	originalInstruction = *originalFunctionPtr;
	if( !err && ((originalInstruction & kMFCTRMask) == kMFCTRInstruction) )
		err = err_cannot_override;
#elif defined(__i386__) || defined(__x86_64__)
	int eatenCount = 0;
	char originalInstructions[kOriginalInstructionsSize];
	uint64_t jumpRelativeInstruction = 0; // JMP

	Boolean overridePossible = eatKnownInstructions ((unsigned char *)originalFunctionPtr, 
										&jumpRelativeInstruction, &eatenCount, originalInstructions);
#ifdef DEBUG_DISASM
  if (!overridePossible) fprintf(stderr, "overridePossible = false @%d\n", __LINE__);
#endif
	if (eatenCount > kOriginalInstructionsSize) {
#ifdef DEBUG_DISASM
		fprintf(stderr, "Too many instructions eaten\n");
#endif    
		overridePossible = false;
	}
	if (!overridePossible) err = err_cannot_override;
	if (err) printf("err = %x %d\n", err, __LINE__);
#endif
	
	//	Make the original function implementation writable.
	if( !err ) {
		err = vm_protect( mach_task_self(),
				(vm_address_t) originalFunctionPtr,
				sizeof(long), false, (VM_PROT_ALL | VM_PROT_COPY) );
		if( err )
			err = vm_protect( mach_task_self(),
					(vm_address_t) originalFunctionPtr, sizeof(long), false,
					(VM_PROT_DEFAULT | VM_PROT_COPY) );
	}
	if (err) printf("err = %x %d\n", err, __LINE__);
	
	//	Allocate and target the escape island to the overriding function.
	BranchIsland	*escapeIsland = NULL;
	if( !err )	
		err = allocateBranchIsland( &escapeIsland, kAllocateHigh, originalFunctionAddress );
		if (err) printf("err = %x %d\n", err, __LINE__);

	
#if defined(__ppc__) || defined(__POWERPC__)
	if( !err )
		err = setBranchIslandTarget( escapeIsland, overrideFunctionAddress, 0 );
	
	//	Build the branch absolute instruction to the escape island.
	long	branchAbsoluteInstruction = 0; // Set to 0 just to silence warning.
	if( !err ) {
		long escapeIslandAddress = ((long) escapeIsland) & 0x3FFFFFF;
		branchAbsoluteInstruction = 0x48000002 | escapeIslandAddress;
	}
#elif defined(__i386__) || defined(__x86_64__)
        if (err) printf("err = %x %d\n", err, __LINE__);

	if( !err )
		err = setBranchIslandTarget_i386( escapeIsland, overrideFunctionAddress, 0 );
 
	if (err) printf("err = %x %d\n", err, __LINE__);
	// Build the jump relative instruction to the escape island
#endif


#if defined(__i386__) || defined(__x86_64__)
	if (!err) {
		uint32_t addressOffset = ((char*)escapeIsland - (char*)originalFunctionPtr - 5);
		addressOffset = OSSwapInt32(addressOffset);
		
		jumpRelativeInstruction |= 0xE900000000000000LL; 
		jumpRelativeInstruction |= ((uint64_t)addressOffset & 0xffffffff) << 24;
		jumpRelativeInstruction = OSSwapInt64(jumpRelativeInstruction);		
	}
#endif
	
	//	Optionally allocate & return the reentry island.
	BranchIsland	*reentryIsland = NULL;
	if( !err && originalFunctionReentryIsland ) {
		err = allocateBranchIsland( &reentryIsland, kAllocateNormal, NULL);
		if( !err )
			*originalFunctionReentryIsland = reentryIsland;
	}
	
#if defined(__ppc__) || defined(__POWERPC__)	
	//	Atomically:
	//	o If the reentry island was allocated:
	//		o Insert the original instruction into the reentry island.
	//		o Target the reentry island at the 2nd instruction of the
	//		  original function.
	//	o Replace the original instruction with the branch absolute.
	if( !err ) {
		int escapeIslandEngaged = false;
		do {
			if( reentryIsland )
				err = setBranchIslandTarget( reentryIsland,
						(void*) (originalFunctionPtr+1), originalInstruction );
			if( !err ) {
				escapeIslandEngaged = CompareAndSwap( originalInstruction,
										branchAbsoluteInstruction,
										(UInt32*)originalFunctionPtr );
				if( !escapeIslandEngaged ) {
					//	Someone replaced the instruction out from under us,
					//	re-read the instruction, make sure it's still not
					//	'mfctr' and try again.
					originalInstruction = *originalFunctionPtr;
					if( (originalInstruction & kMFCTRMask) == kMFCTRInstruction)
						err = err_cannot_override;
				}
			}
		} while( !err && !escapeIslandEngaged );
	}
#elif defined(__i386__) || defined(__x86_64__)
	// Atomically:
	//	o If the reentry island was allocated:
	//		o Insert the original instructions into the reentry island.
	//		o Target the reentry island at the first non-replaced 
	//        instruction of the original function.
	//	o Replace the original first instructions with the jump relative.
	//
	// Note that on i386, we do not support someone else changing the code under our feet
	if ( !err ) {
		if( reentryIsland )
			err = setBranchIslandTarget_i386( reentryIsland,
										 (void*) ((char *)originalFunctionPtr+eatenCount), originalInstructions );
		if ( !err )
			atomic_mov64((uint64_t *)originalFunctionPtr, jumpRelativeInstruction);
	}
#endif
	
	//	Clean up on error.
	if( err ) {
		if( reentryIsland )
			freeBranchIsland( reentryIsland );
		if( escapeIsland )
			freeBranchIsland( escapeIsland );
	}

#if defined(__x86_64__)
        err = makeIslandExecutable(escapeIsland);
        err = makeIslandExecutable(reentryIsland);
#endif
#ifdef DEBUG_DISASM
  {
    fprintf(stderr, "First 16 bytes of the function after slicing: ");
    unsigned char *orig = (unsigned char *)originalFunctionAddress;
    int i;
    for (i = 0; i < 16; i++) {
       fprintf(stderr, "%x ", (unsigned int) orig[i]);
    }
    fprintf(stderr, "\n");
  }
#endif

	return err;
}

/*******************************************************************************
*	
*	Implementation
*	
*******************************************************************************/
#pragma mark	-
#pragma mark	(Implementation)

/***************************************************************************//**
	Implementation: Allocates memory for a branch island.
	
	@param	island			<-	The allocated island.
	@param	allocateHigh	->	Whether to allocate the island at the end of the
								address space (for use with the branch absolute
								instruction).
	@result					<-	mach_error_t

	***************************************************************************/

	mach_error_t
allocateBranchIsland(
		BranchIsland	**island,
		int				allocateHigh,
		void *originalFunctionAddress)
{
	assert( island );
	
	mach_error_t	err = err_none;
	
	if( allocateHigh ) {
		vm_size_t pageSize;
		err = host_page_size( mach_host_self(), &pageSize );
		if( !err ) {
			assert( sizeof( BranchIsland ) <= pageSize );
#if defined(__x86_64__)
			vm_address_t first = (uint64_t)originalFunctionAddress & ~(uint64_t)(((uint64_t)1 << 31) - 1) | ((uint64_t)1 << 31); // start in the middle of the page?
			vm_address_t last = 0x0;
#else
			vm_address_t first = 0xfeffffff;
			vm_address_t last = 0xfe000000 + pageSize;
#endif

			vm_address_t page = first;
			int allocated = 0;
			vm_map_t task_self = mach_task_self();
			
			while( !err && !allocated && page != last ) {

				err = vm_allocate( task_self, &page, pageSize, 0 );
				if( err == err_none )
					allocated = 1;
				else if( err == KERN_NO_SPACE ) {
#if defined(__x86_64__)
					page -= pageSize;
#else
					page += pageSize;
#endif
					err = err_none;
				}
			}
			if( allocated )
				*island = (BranchIsland*) page;
			else if( !allocated && !err )
				err = KERN_NO_SPACE;
		}
	} else {
		void *block = malloc( sizeof( BranchIsland ) );
		if( block )
			*island = (BranchIsland*)block;
		else
			err = KERN_NO_SPACE;
	}
	if( !err )
		(**island).allocatedHigh = allocateHigh;
	
	return err;
}

/***************************************************************************//**
	Implementation: Deallocates memory for a branch island.
	
	@param	island	->	The island to deallocate.
	@result			<-	mach_error_t

	***************************************************************************/

	mach_error_t
freeBranchIsland(
		BranchIsland	*island )
{
	assert( island );
	assert( (*(long*)&island->instructions[0]) == kIslandTemplate[0] );
	assert( island->allocatedHigh );
	
	mach_error_t	err = err_none;
	
	if( island->allocatedHigh ) {
		vm_size_t pageSize;
		err = host_page_size( mach_host_self(), &pageSize );
		if( !err ) {
			assert( sizeof( BranchIsland ) <= pageSize );
			err = vm_deallocate(
					mach_task_self(),
					(vm_address_t) island, pageSize );
		}
	} else {
		free( island );
	}
	
	return err;
}

/***************************************************************************//**
	Implementation: Sets the branch island's target, with an optional
	instruction.
	
	@param	island		->	The branch island to insert target into.
	@param	branchTo	->	The address of the target.
	@param	instruction	->	Optional instruction to execute prior to branch. Set
							to zero for nop.
	@result				<-	mach_error_t

	***************************************************************************/
#if defined(__ppc__) || defined(__POWERPC__)
	mach_error_t
setBranchIslandTarget(
		BranchIsland	*island,
		const void		*branchTo,
		long			instruction )
{
	//	Copy over the template code.
    bcopy( kIslandTemplate, island->instructions, sizeof( kIslandTemplate ) );
    
    //	Fill in the address.
    ((short*)island->instructions)[kAddressLo] = ((long) branchTo) & 0x0000FFFF;
    ((short*)island->instructions)[kAddressHi]
    	= (((long) branchTo) >> 16) & 0x0000FFFF;
    
    //	Fill in the (optional) instuction.
    if( instruction != 0 ) {
        ((short*)island->instructions)[kInstructionLo]
        	= instruction & 0x0000FFFF;
        ((short*)island->instructions)[kInstructionHi]
        	= (instruction >> 16) & 0x0000FFFF;
    }
    
    //MakeDataExecutable( island->instructions, sizeof( kIslandTemplate ) );
	msync( island->instructions, sizeof( kIslandTemplate ), MS_INVALIDATE );
    
    return err_none;
}
#endif 

#if defined(__i386__)
	mach_error_t
setBranchIslandTarget_i386(
	BranchIsland	*island,
	const void		*branchTo,
	char*			instructions )
{

	//	Copy over the template code.
    bcopy( kIslandTemplate, island->instructions, sizeof( kIslandTemplate ) );

	// copy original instructions
	if (instructions) {
		bcopy (instructions, island->instructions + kInstructions, kOriginalInstructionsSize);
	}
	
    // Fill in the address.
    int32_t addressOffset = (char *)branchTo - (island->instructions + kJumpAddress + 4);
    *((int32_t *)(island->instructions + kJumpAddress)) = addressOffset; 

    msync( island->instructions, sizeof( kIslandTemplate ), MS_INVALIDATE );
    return err_none;
}

#elif defined(__x86_64__)
mach_error_t
setBranchIslandTarget_i386(
        BranchIsland	*island,
        const void		*branchTo,
        char*			instructions )
{
    // Copy over the template code.
    bcopy( kIslandTemplate, island->instructions, sizeof( kIslandTemplate ) );

    // Copy original instructions.
    if (instructions) {
        bcopy (instructions, island->instructions, kOriginalInstructionsSize);
    }

    //	Fill in the address.
    *((uint64_t *)(island->instructions + kJumpAddress)) = (uint64_t)branchTo; 
    msync( island->instructions, sizeof( kIslandTemplate ), MS_INVALIDATE );

    return err_none;
}
#endif


#if defined(__i386__) || defined(__x86_64__)
// simplistic instruction matching
typedef struct {
	unsigned int length; // max 15
	unsigned char mask[15]; // sequence of bytes in memory order
	unsigned char constraint[15]; // sequence of bytes in memory order
}	AsmInstructionMatch;

#if defined(__i386__)
static AsmInstructionMatch possibleInstructions[] = {
	{ 0x1, {0xFF}, {0x90} },							// nop
	{ 0x1, {0xFF}, {0x55} },							// push %esp
	{ 0x2, {0xFF, 0xFF}, {0x89, 0xE5} },				                // mov %esp,%ebp
	{ 0x3, {0xFF, 0xFF, 0xFF}, {0x89, 0x1C, 0x24} },				                // mov %ebx,(%esp)
	{ 0x1, {0xFF}, {0x53} },							// push %ebx
	{ 0x3, {0xFF, 0xFF, 0x00}, {0x83, 0xEC, 0x00} },	                        // sub 0x??, %esp
	{ 0x6, {0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00}, {0x81, 0xEC, 0x00, 0x00, 0x00, 0x00} },	// sub 0x??, %esp with 32bit immediate
	{ 0x1, {0xFF}, {0x57} },							// push %edi
	{ 0x1, {0xFF}, {0x56} },							// push %esi
	{ 0x2, {0xFF, 0xFF}, {0x31, 0xC0} },						// xor %eax, %eax
	{ 0x3, {0xFF, 0x4F, 0x00}, {0x8B, 0x45, 0x00} },  // mov $imm(%ebp), %reg
	{ 0x3, {0xFF, 0x4C, 0x00}, {0x8B, 0x40, 0x00} },  // mov $imm(%eax-%edx), %reg
        { 0x4, {0xFF, 0x00, 0x00, 0x00}, {0x8B, 0x00, 0x00, 0x00} },  // mov r16,r/m16 or r32,r/m32
        { 0x5, {0xFF, 0x00, 0x00, 0x00, 0x00}, {0xB9, 0x00, 0x00, 0x00, 0x00} }, // mov $imm, %ecx
  { 0x5, {0xFF, 0x00, 0x00, 0x00, 0x00}, {0xB8, 0x00, 0x00, 0x00, 0x00} }, // mov $imm, %eax
        { 0x4, {0xFF, 0xFF, 0xFF, 0x00}, {0x66, 0x0F, 0xEF, 0x00} },             // pxor xmm2/128, xmm1
        { 0x2, {0xFF, 0xFF}, {0xDB, 0xE3} }, // fninit
	{ 0x0 }
};
#elif defined(__x86_64__)
// TODO(glider): disassembling the "0x48, 0x89" sequences is trickier than it's done below.
// If it stops working, refer to http://ref.x86asm.net/geek.html#modrm_byte_32_64 to do it
// more accurately.
static AsmInstructionMatch possibleInstructions[] = {
	{ 0x1, {0xFF}, {0x90} },							// nop
	{ 0x1, {0xF8}, {0x50} },							// push %rX
  { 0x1, {0xFF}, {0x65} },              // GS prefix
	{ 0x3, {0xFF, 0xFF, 0xFF}, {0x48, 0x89, 0xE5} },				// mov %rsp,%rbp
	{ 0x4, {0xFF, 0xFF, 0xFF, 0x00}, {0x48, 0x83, 0xEC, 0x00} },	                // sub 0x??, %rsp
	{ 0x4, {0xFB, 0xFF, 0x07, 0x00}, {0x48, 0x89, 0x05, 0x00} },	                // move onto rbp
	{ 0x3, {0xFB, 0xFF, 0x00}, {0x48, 0x89, 0x00} },	                            // mov %reg, %reg
	{ 0x2, {0xFF, 0x00}, {0x41, 0x00} },						// push %rXX
	{ 0x2, {0xFF, 0x00}, {0x85, 0x00} },						// test %rX,%rX
	{ 0x2, {0xFF, 0x00}, {0x77, 0x00} },						// ja $i8
	{ 0x5, {0xF8, 0x00, 0x00, 0x00, 0x00}, {0xB8, 0x00, 0x00, 0x00, 0x00} },        // mov $imm, %reg
  { 0x5, {0xFF, 0x00, 0x00, 0x00, 0x00}, {0x25, 0x00, 0x00, 0x00, 0x00} },  // and $imm, %eax

  { 0x8, {0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00},
         {0x48, 0x8B, 0x34, 0x25, 0x00, 0x00, 0x00, 0x00}, },                     // mov $imm, %rsi
  { 0x4, {0xFF, 0xFF, 0xFF, 0x00}, {0x48, 0x83, 0xFA, 0x00}, },   // cmp $i8, %rdx
	{ 0xa, {0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
               {0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00} },    // mov $imm, %rax
        { 0x6, {0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00},
               {0x81, 0xE6, 0x00, 0x00, 0x00, 0x00} },                            // and $imm, %esi
        { 0x4, {0xFF, 0xFF, 0xFF, 0x00}, {0x66, 0x0F, 0xEF, 0x00} },              // pxor xmm2/128, xmm1
        { 0x2, {0xFF, 0x00}, {0x89, 0x00} },                               // mov r/m32,r32 or r/m16,r16
        { 0x3, {0xFF, 0xFF, 0xFF}, {0x49, 0x89, 0xF8} },                   // mov %rdi,%r8        
        { 0x3, {0xFF, 0xFF, 0x00}, {0xFF, 0x77, 0x00} },  // pushq $imm(%rdi)
        { 0x2, {0xFF, 0xFF}, {0xDB, 0xE3} }, // fninit
	{ 0x0 }
};
#endif

static Boolean codeMatchesInstruction(unsigned char *code, AsmInstructionMatch* instruction)
{
	Boolean match = true;
	size_t i;
  assert(instruction);
#ifdef DEBUG_DISASM
  fprintf(stderr, "Matching: ");
#endif  
	for (i=0; i<instruction->length; i++) {
		unsigned char mask = instruction->mask[i];
		unsigned char constraint = instruction->constraint[i];
		unsigned char codeValue = code[i];
#ifdef DEBUG_DISASM
    fprintf(stderr, "%x ", (unsigned)codeValue);
#endif    
		match = ((codeValue & mask) == constraint);
		if (!match) break;
	}
#ifdef DEBUG_DISASM
  if (match) {
    fprintf(stderr, " OK\n");
  } else {
    fprintf(stderr, " FAIL\n");
  }
#endif  
	return match;
}

#if defined(__i386__) || defined(__x86_64__)
	static Boolean 
eatKnownInstructions( 
	unsigned char *code, 
	uint64_t* newInstruction,
	int* howManyEaten, 
	char* originalInstructions )
{
	Boolean allInstructionsKnown = true;
	int totalEaten = 0;
	unsigned char* ptr = code;
	int remainsToEat = 5; // a JMP instruction takes 5 bytes
	
	if (howManyEaten) *howManyEaten = 0;
	while (remainsToEat > 0) {
		Boolean curInstructionKnown = false;
		
		// See if instruction matches one  we know
		AsmInstructionMatch* curInstr = possibleInstructions;
		do { 
			if ((curInstructionKnown = codeMatchesInstruction(ptr, curInstr))) break;
			curInstr++;
		} while (curInstr->length > 0);
		
		// if all instruction matches failed, we don't know current instruction then, stop here
		if (!curInstructionKnown) { 
			allInstructionsKnown = false;
			break;
		}
		
		// At this point, we've matched curInstr
		int eaten = curInstr->length;
		ptr += eaten;
		remainsToEat -= eaten;
		totalEaten += eaten;
	}


	if (howManyEaten) *howManyEaten = totalEaten;

	if (originalInstructions) {
		Boolean enoughSpaceForOriginalInstructions = (totalEaten < kOriginalInstructionsSize);
		
		if (enoughSpaceForOriginalInstructions) {
			memset(originalInstructions, 0x90 /* NOP */, kOriginalInstructionsSize); // fill instructions with NOP
			bcopy(code, originalInstructions, totalEaten);
		} else {
#ifdef DEBUG_DISASM    
			fprintf (stderr, "Not enough space in island to store original instructions. Adapt the island definition and kOriginalInstructionsSize\n");
#endif      
			return false;
		}
	}
	
	if (allInstructionsKnown) {
		// save last 3 bytes of first 64bits of codre we'll replace
		uint64_t currentFirst64BitsOfCode = *((uint64_t *)code);
		currentFirst64BitsOfCode = OSSwapInt64(currentFirst64BitsOfCode); // back to memory representation
		currentFirst64BitsOfCode &= 0x0000000000FFFFFFLL; 
		
		// keep only last 3 instructions bytes, first 5 will be replaced by JMP instr
		*newInstruction &= 0xFFFFFFFFFF000000LL; // clear last 3 bytes
		*newInstruction |= (currentFirst64BitsOfCode & 0x0000000000FFFFFFLL); // set last 3 bytes
	}

	return allInstructionsKnown;
}
#endif

#if defined(__i386__)
__asm(
			".text;"
			".align 2, 0x90;"
			"_atomic_mov64:;"
			"	pushl %ebp;"
			"	movl %esp, %ebp;"
			"	pushl %esi;"
			"	pushl %ebx;"
			"	pushl %ecx;"
			"	pushl %eax;"
			"	pushl %edx;"
	
			// atomic push of value to an address
			// we use cmpxchg8b, which compares content of an address with 
			// edx:eax. If they are equal, it atomically puts 64bit value 
			// ecx:ebx in address. 
			// We thus put contents of address in edx:eax to force ecx:ebx
			// in address
			"	mov		8(%ebp), %esi;"  // esi contains target address
			"	mov		12(%ebp), %ebx;"
			"	mov		16(%ebp), %ecx;" // ecx:ebx now contains value to put in target address
			"	mov		(%esi), %eax;"
			"	mov		4(%esi), %edx;"  // edx:eax now contains value currently contained in target address
			"	lock; cmpxchg8b	(%esi);" // atomic move.
			
			// restore registers
			"	popl %edx;"
			"	popl %eax;"
			"	popl %ecx;"
			"	popl %ebx;"
			"	popl %esi;"
			"	popl %ebp;"
			"	ret"
);
#elif defined(__x86_64__)
void atomic_mov64(
		uint64_t *targetAddress,
		uint64_t value )
{
    *targetAddress = value;
}
#endif
#endif
