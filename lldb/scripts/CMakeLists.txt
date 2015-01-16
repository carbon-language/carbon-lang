set(LLVM_NO_RTTI 1)

file(GLOB SWIG_INPUTS Python/interface/*.i)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    STRING(REGEX REPLACE ".lib" ".dll" PYTHON_DLL ${PYTHON_LIBRARY})
    file(COPY ${PYTHON_DLL} DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif ()

if ( LLDB_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION )
	find_package(SWIG REQUIRED)
	add_custom_command(
	  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
	  DEPENDS ${LLDB_SOURCE_DIR}/scripts/lldb.swig
	  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/buildSwigWrapperClasses.py
	  DEPENDS ${SWIG_INPUTS}
	  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/buildSwigWrapperClasses.py "--srcRoot=${LLDB_SOURCE_DIR}" "--targetDir=${CMAKE_CURRENT_BINARY_DIR}" "--cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}" "--prefix=${CMAKE_BINARY_DIR}" "--swigExecutable=${SWIG_EXECUTABLE}" -m
	  COMMENT "Python script building LLDB Python wrapper")  
	set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp PROPERTIES GENERATED 1)

	add_custom_target(swig_wrapper ALL
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
		)

	# Install the LLDB python module on all operating systems (except Windows)
	if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
	  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
		DESTINATION lib${LLVM_LIBDIR_SUFFIX})
	endif()
else ()
	add_custom_command(
	  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
	  DEPENDS ${LLDB_SOURCE_DIR}/scripts/lldb.swig
	  DEPENDS ${SWIG_INPUTS}
	  # swig was directly invoked on Windows (where the Python API is not being generated) but on other platforms, we need to run the *swig-wrapper-classes.sh shell-scripts.
	  #COMMAND swig -c++ -shadow -python -I${LLDB_SOURCE_DIR}/include -I./. -outdir ${LLDB_SOURCE_DIR}/scripts/Python  -o ${LLDB_SOURCE_DIR}/source/LLDBWrapPython.cpp ${LLDB_SOURCE_DIR}/scripts/lldb.swig
	  COMMAND env PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/build-swig-wrapper-classes.sh ${LLDB_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR} -m
	  COMMAND env PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/finish-swig-wrapper-classes.sh ${LLDB_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR} -m
	  COMMENT "Building lldb python wrapper")
	set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp PROPERTIES GENERATED 1)

	add_custom_target(swig_wrapper ALL
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
		)

	# Install the LLDB python module on all operating systems (except Windows)
	if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
	  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
		DESTINATION lib${LLVM_LIBDIR_SUFFIX})
	endif()
endif ()

# build Python modules
add_subdirectory(Python/modules)
