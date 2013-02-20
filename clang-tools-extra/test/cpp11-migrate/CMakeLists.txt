
# List of generator scripts. Generator scripts must:
# * Be written in python
# * Output their result to standard out.
# * Be named as gen_X.py where X will be the name of the auto-generated file.
set(generator_scripts
  UseAuto/gen_basic_std_iterator_tests.cpp.py
  UseAuto/Inputs/gen_my_std.h.py
  )

# macro that runs a generator script to produce an auto-generated file.
# Generator scripts must follow scheme above. The resulting file is placed in:
# ${CMAKE_CURRENT_BINARY_DIR})/generated_tests/dirname(<script name>).
#
# Two arguments are required:
# script - The generator script (relative to ${CMAKE_CURRENT_SOURCE_DIR})
# output - Name of the variable to store the name of the generated file in.
macro(autogenerate_file script output)
  get_filename_component(dir ${script} PATH)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_tests/${dir}")
  string(REGEX REPLACE "gen_(.*).py" "\\1" output_name ${script})
  set(script_file
    "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
    )
  set(output_file
    "${CMAKE_CURRENT_BINARY_DIR}/generated_tests/${output_name}"
    )
  set(gencmd
    ${PYTHON_EXECUTABLE}
    ${script_file} > ${output_file}
    )
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${gencmd}
    DEPENDS ${script_file}
    )
  set(${output} ${output_file})
endmacro(autogenerate_file)

# Define rules to run generator scripts.
set(depends)
foreach(script ${generator_scripts})
  autogenerate_file(${script} output)
  list(APPEND depends ${output})
endforeach()

# This custom target will be used by parent CMakeLists.txt to ensure all
# auto-generated files are created before lit is run.
add_custom_target(cpp11-migrate-autogen
  DEPENDS ${depends}
  )

set(TEST_SOURCE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/generated_tests)
set(TEST_EXEC_ROOT ${CMAKE_CURRENT_BINARY_DIR})
set(TESTSUITE_NAME "cpp11-migrate Auto-Generated Tests")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated_tests/lit.site.cfg
  )
