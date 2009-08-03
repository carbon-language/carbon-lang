PROJECT( timing )

SET( SRCS
 lshrdi3.c
 floatundixf.c
 floatdixf.c
 umoddi3.c
 udivdi3.c
 negdi2.c
 ashrdi3.c
 muldi3.c
 ashldi3.c
 divdi3.c
 floatundisf.c
 floatdidf.c
 floatdisf.c
 moddi3.c
 floatundidf.c
 )

ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SRCS})
# ADD_EXECUTABLE( ${PROJECT_NAME} ${SRCS} )
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${LIBS} )
