add_subdirectory(BrainF)
add_subdirectory(Fibonacci)
add_subdirectory(HowToUseJIT)
add_subdirectory(Kaleidoscope)
add_subdirectory(ModuleMaker)

if( NOT WIN32 )
    add_subdirectory(ExceptionDemo)
endif()

include(CheckIncludeFile)
check_include_file(pthread.h HAVE_PTHREAD_H)

if( HAVE_PTHREAD_H )
  add_subdirectory(ParallelJIT)
endif( HAVE_PTHREAD_H )
