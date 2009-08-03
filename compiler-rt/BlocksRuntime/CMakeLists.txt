PROJECT( BlocksRuntime )

SET( SRCS
 runtime.c
 data.c
 )

ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SRCS})
