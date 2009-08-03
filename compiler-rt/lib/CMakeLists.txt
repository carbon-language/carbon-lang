#
# Create a library called "CompilerRT" which includes the source files.

# Generic functions needed for each architecture

SET( SRCS
 absvdi2.c               cmpdi2.c                fixdfdi.c               floatdidf.c             moddi3.c                negvti2.c               ucmpdi2.c
 absvsi2.c               cmpti2.c                fixdfti.c               floatdisf.c             modsi3.c                paritydi2.c             ucmpti2.c
 absvti2.c               ctzdi2.c                fixsfdi.c               floatdixf.c             modti3.c                paritysi2.c             udivdi3.c
 addvdi3.c               ctzsi2.c                fixsfti.c               floattidf.c             muldc3.c                parityti2.c             udivmoddi4.c
 addvsi3.c               ctzti2.c                fixunsdfdi.c            floattisf.c             muldi3.c                popcountdi2.c           udivmodti4.c
 addvti3.c               divdc3.c                fixunsdfsi.c            floattixf.c             mulsc3.c                popcountsi2.c           udivsi3.c
 apple_versioning.c      divdi3.c                fixunsdfti.c            floatundidf.c           multi3.c                popcountti2.c           udivti3.c
 ashldi3.c               divsc3.c                fixunssfdi.c            floatundisf.c           mulvdi3.c               powidf2.c               umoddi3.c
 ashlti3.c               divsi3.c                fixunssfsi.c            floatundixf.c           mulvsi3.c               powisf2.c               umodsi3.c
 ashrdi3.c               divti3.c                fixunssfti.c            floatuntidf.c           mulvti3.c               powitf2.c               umodti3.c
 ashrti3.c               divxc3.c                fixunsxfdi.c            floatuntisf.c           mulxc3.c                powixf2.c
 clear_cache.c           enable_execute_stack.c  fixunsxfsi.c            floatuntixf.c           negdi2.c                subvdi3.c
 clzdi2.c                eprintf.c               fixunsxfti.c            gcc_personality_v0.c    negti2.c                subvsi3.c
 clzsi2.c                ffsdi2.c                fixxfdi.c               lshrdi3.c               negvdi2.c               subvti3.c
 clzti2.c                ffsti2.c                fixxfti.c               lshrti3.c               negvsi2.c               trampoline_setup.c
 )

# Optimized functions for each architecture

# Commenting out for the min until the basics are working first.
# ADD_SUBDIRECTORY( ppc )
# ADD_SUBDIRECTORY( x86_64 )
# ADD_SUBDIRECTORY( i386 )

# Creates a shared lib .so
ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SRCS} )
