# Test target to run Python test suite from main build.

add_custom_target(check-clang-python
	COMMAND CLANG_LIBRARY_PATH=$<TARGET_FILE_DIR:libclang> ${PYTHON_EXECUTABLE} -m unittest discover
	DEPENDS libclang
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_dependencies(check-all check-clang-python)
