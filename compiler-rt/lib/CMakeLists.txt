#
# Create a library called "CompilerRT" which includes the source files.

#INCLUDE_DIRECTORIES(
# ${CMAKE_CURRENT_BINARY_DIR}
#)

# Generic functions needed for each architecture
# libcompiler_rt.Generic.a    libcompiler_rt.Optimized.a

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

# Generic
SET( Common_SRCS
 absvdi2.c		absvsi2.c		addvdi3.c		addvsi3.c		ashldi3.c		ashrdi3.c
 clear_cache.c		clzdi2.c		clzsi2.c		cmpdi2.c		ctzdi2.c		ctzsi2.c
 divdc3.c		divdi3.c		divsc3.c		enable_execute_stack.c	ffsdi2.c
 fixdfdi.c		fixsfdi.c		fixunsdfdi.c		fixunsdfsi.c		fixunssfdi.c
 fixunssfsi.c		floatdidf.c		floatdisf.c		floatundidf.c		floatundisf.c
 gcc_personality_v0.c	lshrdi3.c		moddi3.c		muldc3.c		muldi3.c
 mulsc3.c		mulvdi3.c		mulvsi3.c		negdi2.c		negvdi2.c		negvsi2.c
 paritydi2.c		paritysi2.c		popcountdi2.c		popcountsi2.c		powidf2.c
 powisf2.c		subvdi3.c		subvsi3.c		ucmpdi2.c		udivdi3.c
 udivmoddi4.c		umoddi3.c		apple_versioning.c	eprintf.c
 )

SET( i386_Functions
 divxc3.c fixunsxfdi.c fixunsxfsi.c fixxfdi.c
 floatdixf.c floatundixf.c mulxc3.c powixf2.c
 )

# Optimized functions for each architecture

# Commenting out for the min until the basics are working first.
# ADD_SUBDIRECTORY( ppc )
# ADD_SUBDIRECTORY( x86_64 )
# ADD_SUBDIRECTORY( i386 )

#FOREACH( LOOP_VAR ${Achitectures} )
# See ARCHIVE_OUTPUT_DIRECTORY docs.
#${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LOOP_VAR}
#ENDFOREACH

ADD_LIBRARY( ${PROJECT_NAME}-Common STATIC ${Common_SRCS} )
#ADD_LIBRARY( ${PROJECT_NAME}-i386 STATIC ${i386_Functions} )

# [[debug|optimized|general]
#TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${PROJECT_NAME}-Common optimized ${PROJECT_NAME}-i386 )

