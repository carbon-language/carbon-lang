#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

####################
# --- Create all ---
add_custom_target(libomp-lib ALL DEPENDS omp)
if(${LIBOMP_FORTRAN_MODULES})
    add_custom_target(libomp-mod ALL DEPENDS ${export_mod_files})
endif()

#############################
# --- Create Common Files ---
add_custom_target(libomp-common ALL DEPENDS ${export_cmn_files})
add_custom_target(libomp-clean-common COMMAND ${CMAKE_COMMAND} -E remove -f ${export_cmn_files})

# --- Put headers in convenient locations post build ---
if(${LIBOMP_COPY_EXPORTS})
    add_custom_command(TARGET common POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E make_directory ${export_cmn_dir}
        COMMAND ${CMAKE_COMMAND} -E copy omp.h ${export_cmn_dir}
        COMMAND ${CMAKE_COMMAND} -E copy omp_lib.h ${export_cmn_dir}
        COMMAND ${CMAKE_COMMAND} -E copy omp_lib.f ${export_cmn_dir}
        COMMAND ${CMAKE_COMMAND} -E copy omp_lib.f90 ${export_cmn_dir}
    )
    if(${LIBOMP_OMPT_SUPPORT})
        add_custom_command(TARGET common POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ompt.h ${export_cmn_dir}
        )
    endif()
    if(${LIBOMP_FORTRAN_MODULES})
        add_custom_command(TARGET mod POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${export_mod_dir}
            COMMAND ${CMAKE_COMMAND} -E copy omp_lib.mod ${export_mod_dir}
            COMMAND ${CMAKE_COMMAND} -E copy omp_lib_kinds.mod ${export_mod_dir}
        )
    endif()
endif()

######################################################
# --- Build the main library ---
# $(lib_file) <== Main library file to create

# preprocessor flags (-D definitions and -I includes)
# Grab environment variable CPPFLAGS and append those to definitions
set(include_dirs ${CMAKE_CURRENT_BINARY_DIR} ${src_dir} ${src_dir}/i18n ${inc_dir} ${src_dir}/thirdparty/ittnotify)
include_directories(${include_dirs})

# objects depend on : .inc files and omp.h
# This way the *.inc and omp.h are generated before any compilations take place
add_custom_target(libomp-needed-headers DEPENDS kmp_i18n_id.inc kmp_i18n_default.inc omp.h)

# For Windows, there is a definitions file (.def) and resource file (.res) created using generate-def.pl and rc.exe respectively.
if(${WINDOWS})
    add_custom_target(libomp-needed-windows-files DEPENDS ${def_file} ${CMAKE_CURRENT_BINARY_DIR}/libomp.rc)
    list(APPEND lib_src_files ${CMAKE_CURRENT_BINARY_DIR}/libomp.rc)
endif()

# Remove any cmake-automatic linking of libraries by linker, This is so linux 
# and mac don't include libstdc++ just because we compile c++ files.
if(${LIBOMP_USE_PREDEFINED_LINKER_FLAGS})
    set(CMAKE_C_IMPLICIT_LINK_LIBRARIES   "")
    set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")
    set(CMAKE_ASM_IMPLICIT_LINK_LIBRARIES "")
endif()

# --- ${lib_file} rule ---
add_library(omp SHARED ${lib_src_files})
set_target_properties(omp PROPERTIES 
    PREFIX "" SUFFIX ""        # Take control 
    OUTPUT_NAME "${lib_file}"  # of output name
    LINK_FLAGS  "${LD_FLAGS}"   
    LINKER_LANGUAGE C          # use C Compiler for linking step
    SKIP_BUILD_RPATH true      # have Mac linker -install_name just be "-install_name libomp.dylib"
)

# --- Copy libomp into exports directory post build ---
if(${WINDOWS})
    get_target_property(LIBOMP_OUTPUT_DIRECTORY omp RUNTIME_OUTPUT_DIRECTORY)
else()
    get_target_property(LIBOMP_OUTPUT_DIRECTORY omp LIBRARY_OUTPUT_DIRECTORY)
endif()
if(NOT LIBOMP_OUTPUT_DIRECTORY)
    set(LIBOMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(${LIBOMP_COPY_EXPORTS})
    add_custom_command(TARGET omp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${export_lib_dir}
        COMMAND ${CMAKE_COMMAND} -E copy ${LIBOMP_OUTPUT_DIRECTORY}/${lib_file} ${export_lib_dir}
    )
endif()

# Linking command will include libraries in LD_LIB_FLAGS
target_link_libraries(omp ${LD_LIB_FLAGS} ${CMAKE_DL_LIBS})

# Create *.inc and omp.h before compiling any sources
add_dependencies(omp libomp-needed-headers)
if(${WINDOWS})
# Create .def and .rc file before compiling any sources
    add_dependencies(omp libomp-needed-windows-files)
endif()

# Set the compiler flags for each type of source
set_source_files_properties(${lib_c_items} 
                            ${imp_c_items} PROPERTIES COMPILE_FLAGS "${C_FLAGS}")
set_source_files_properties(${lib_cxx_items} PROPERTIES COMPILE_FLAGS "${CXX_FLAGS}")
if(${WINDOWS})
    # Windows operating system has to use MASM assembler
    set_source_files_properties(${lib_asm_items} PROPERTIES COMPILE_FLAGS "${ASM_FLAGS}")
else()
    # Non-Windows operating systems can use compiler to assemble the assembly files
    set_source_files_properties(${lib_asm_items} PROPERTIES COMPILE_FLAGS "${ASM_FLAGS}" LANGUAGE C)
endif()
# Set the -D definitions for all sources
add_definitions(${DEFINITIONS_FLAGS})

# If creating a build that imitates build.pl's rules then set USE_BUILDPL_RULES to true
if(${LIBOMP_USE_BUILDPL_RULES})
    include(BuildPLRules)
endif()

######################################################
# --- Source file specific flags ---
# kmp_version.o : -D _KMP_BUILD_TIME="\"$(date)}\""
set_source_files_properties(kmp_version.c  PROPERTIES COMPILE_DEFINITIONS "_KMP_BUILD_TIME=\"\\\"${date}\\\"\"")

if(${WINDOWS})
    set_source_files_properties(thirdparty/ittnotify/ittnotify_static.c PROPERTIES COMPILE_DEFINITIONS "UNICODE")
endif()

######################################################
# MAC specific build rules
if(${MAC})
    # fat library rules
    if(${INTEL64})
        _export_lib_fat_dir("mac_32e" export_fat_mac_32e)
        _export_lib_dir("mac_32"  export_mac_32)
        _export_lib_dir("mac_32e" export_mac_32e)
        add_custom_target(fat
            COMMAND ${CMAKE_COMMAND} -E echo Building 32 and 32e fat libraries from ${export_mac_32}/${lib_file} and ${export_mac_32e}/${lib_file}
            COMMAND ${CMAKE_COMMAND} -E echo Will put fat library in ${export_fat_mac_32e} directory
            COMMAND ${CMAKE_COMMAND} -E make_directory ${export_fat_mac_32e}
            COMMAND lipo -create -output ${export_fat_mac_32e}/${lib_file} ${export_mac_32}/${lib_file} ${export_mac_32e}/${lib_file}
        )
    endif()
endif()

######################################################
# Windows specific build rules
if(${WINDOWS})

    # --- Create $(imp_file) ---
    # This file is first created in the unstripped/${lib_file} creation step.
    # It is then "re-linked" to include kmp_import.c which prevents linking of both Visual Studio OpenMP and newly built OpenMP
    if(NOT "${imp_file}" STREQUAL "")
        set(generated_import_file ${lib_file}${lib})
        add_library(ompimp STATIC ${generated_import_file} ${imp_src_files})
        set_source_files_properties(${generated_import_file} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        set_target_properties(ompimp PROPERTIES
            PREFIX "" SUFFIX ""
            OUTPUT_NAME "${imp_file}"
            STATIC_LIBRARY_FLAGS "${AR_FLAGS}"
            LINKER_LANGUAGE C
            SKIP_BUILD_RPATH true
        )
        add_custom_command(TARGET ompimp PRE_BUILD COMMAND ${CMAKE_COMMAND} -E remove -f ${imp_file})
        add_dependencies(ompimp omp)
        get_target_property(LIBOMPIMP_OUTPUT_DIRECTORY ompimp ARCHIVE_OUTPUT_DIRECTORY)
        if(NOT LIBOMPIMP_OUTPUT_DIRECTORY)
            set(LIBOMPIMP_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif()
        if(${LIBOMP_COPY_EXPORTS})
            add_custom_command(TARGET ompimp POST_BUILD 
                COMMAND ${CMAKE_COMMAND} -E make_directory ${export_lib_dir}
                COMMAND ${CMAKE_COMMAND} -E copy ${LIBOMPIMP_OUTPUT_DIRECTORY}/${imp_file} ${export_lib_dir}
            )
        endif()
    endif()

    # --- Create $(def_file) ---
    if(NOT "${def_file}" STREQUAL "")
        string_to_list("${gd_flags}" gd_flags)
        add_custom_command(
            OUTPUT  ${def_file}
            COMMAND ${PERL_EXECUTABLE} ${tools_dir}/generate-def.pl ${gd_flags} -o ${def_file} ${CMAKE_CURRENT_SOURCE_DIR}/dllexports
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dllexports ${tools_dir}/generate-def.pl
        )
    endif()

endif()

######################################################
# kmp_i18n_id.inc and kmp_i18n_default.inc
set(perlcmd "${PERL_EXECUTABLE}" "${tools_dir}/message-converter.pl" "${oa_opts}" "--prefix=kmp_i18n" "--enum=kmp_i18n_id.inc" "${src_dir}/i18n/en_US.txt")
add_custom_command(
    OUTPUT  kmp_i18n_id.inc
    COMMAND ${perlcmd}
    DEPENDS ${src_dir}/i18n/en_US.txt ${tools_dir}/message-converter.pl
)
set(perlcmd "${PERL_EXECUTABLE}" "${tools_dir}/message-converter.pl" "${oa_opts}" "--prefix=kmp_i18n" "--default=kmp_i18n_default.inc" "${src_dir}/i18n/en_US.txt")
add_custom_command(
    OUTPUT  kmp_i18n_default.inc
    COMMAND ${perlcmd}
    DEPENDS ${src_dir}/i18n/en_US.txt ${tools_dir}/message-converter.pl
)

######################################################
# Micro test rules for after library has been built (cmake/MicroTests.cmake)
# - Only perform if ${LIBOMP_MICRO_TESTS} == true (specify when invoking: cmake -Dtests=on ...)
if(${LIBOMP_MICRO_TESTS})
    include(MicroTests)
    add_custom_target(libomp-micro-tests)
    if(NOT ${MIC} AND ${LIBOMP_TEST_TOUCH})
        add_dependencies(libomp-micro-tests libomp-test-touch)
    endif()
    if(${LINUX} AND ${LIBOMP_TEST_RELO})
        add_dependencies(libomp-micro-tests libomp-test-relo)
    endif()
    if(${LINUX} AND ${LIBOMP_TEST_EXECSTACK})
        add_dependencies(libomp-micro-tests libomp-test-execstack)
    endif()
    if(${MIC} AND ${LIBOMP_TEST_INSTR})
        add_dependencies(libomp-micro-tests libomp-test-instr)
    endif()
    if(${LIBOMP_TEST_DEPS}) 
        add_dependencies(libomp-micro-tests libomp-test-deps)
    endif()
endif()

######################################################
# --- Create Fortran Files ---
# omp_lib.mod
if(${LIBOMP_FORTRAN_MODULES})
    # Grab fortran-compiler-dependent flags
    # Cmake will look for cmake/${CMAKE_Fortran_COMPILER_ID}/FortranFlags.cmake to append additional fortran flags.
    enable_language(Fortran)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_Fortran_COMPILER_ID} ${CMAKE_MODULE_PATH})
    find_file(fortran_specific_include_file_found FortranFlags.cmake ${CMAKE_MODULE_PATH})
    if(fortran_specific_include_file_found)
        include(FortranFlags)
        append_fortran_compiler_specific_fort_flags(F_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_Fortran_COMPILER_ID}/FortranFlags.cmake: will only use default flags in CommonFlags.cmake")
    endif()
    set(omp_lib_f "omp_lib.f90" )
    add_custom_command(
        OUTPUT "omp_lib.mod"
        COMMAND ${CMAKE_Fortran_COMPILER} -c ${F_FLAGS} ${omp_lib_f}
        DEPENDS ${omp_lib_f}
    )
    add_custom_command(
        OUTPUT "omp_lib_kinds.mod"
        COMMAND ${CMAKE_Fortran_COMPILER} -c ${F_FLAGS} ${omp_lib_f}
        DEPENDS ${omp_lib_f}
    )
    # clean omp_lib.o from build directory when "make clean" 
    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES omp_lib${obj})
endif()

###############################################################
# --- Using expand-vars.pl to generate files ---
# - 'file' is generated using expand-vars.pl and 'file'.var
# - Any .h .f .f90 .rc files should be created with this recipe
macro(expand_vars_recipe file_dir filename)
    get_source_file_property(extra_ev_flags ${filename} EV_COMPILE_DEFINITIONS)
    if("${extra_ev_flags}" MATCHES "NOTFOUND")
        set(extra_ev_flags)
    else()
        string_to_list("${extra_ev_flags}" extra_ev_flags)
    endif()
    if(NOT "${filename}" STREQUAL "")
        add_custom_command(
            OUTPUT  ${filename}
            COMMAND ${PERL_EXECUTABLE} ${tools_dir}/expand-vars.pl --strict ${ev_flags} ${extra_ev_flags} ${file_dir}/${filename}.var ${filename}
            DEPENDS ${file_dir}/${filename}.var kmp_version.c ${tools_dir}/expand-vars.pl
        )
    endif()
endmacro()
string_to_list("${ev_flags}" ev_flags)
# omp_lib.h  : ev-flags += -D KMP_INT_PTR_KIND="int_ptr_kind()"
set_source_files_properties(omp_lib.h PROPERTIES EV_COMPILE_DEFINITIONS "-D KMP_INT_PTR_KIND=\"int_ptr_kind()\"") 
# libomp.rc : ev-flags += -D KMP_FILE=$(lib_file)
set_source_files_properties(libomp.rc PROPERTIES EV_COMPILE_DEFINITIONS "-D KMP_FILE=${lib_file}") 
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBOMP_OMP_VERSION} omp.h)
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBOMP_OMP_VERSION} ompt.h)
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBOMP_OMP_VERSION} omp_lib.h)
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBOMP_OMP_VERSION} omp_lib.f)
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBOMP_OMP_VERSION} omp_lib.f90)
expand_vars_recipe(${CMAKE_CURRENT_SOURCE_DIR} libomp.rc)

####################################################################
# Install rules
# We want to install libomp in DESTDIR/CMAKE_INSTALL_PREFIX/lib
# We want to install headers in DESTDIR/CMAKE_INSTALL_PREFIX/include
if(${LIBOMP_STANDALONE_BUILD})
    set(LIBOMP_HEADERS_INSTALL_PATH include)
else()
    string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION ${PACKAGE_VERSION})
    set(LIBOMP_HEADERS_INSTALL_PATH lib${LIBOMP_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)
endif()
if(${WINDOWS})
    install(TARGETS omp RUNTIME DESTINATION bin)
    if(NOT "${imp_file}" STREQUAL "")
        install(TARGETS ompimp ARCHIVE DESTINATION lib${LIBOMP_LIBDIR_SUFFIX})
    endif()
    # Create aliases (regular copies) of the library for backwards compatibility
    set(LIBOMP_ALIASES "libiomp5md")
    foreach(alias IN LISTS LIBOMP_ALIASES)
        install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${lib_file}\" \"${alias}${dll}\"
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin)")
        install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E copy \"${imp_file}\" \"${alias}${lib}\"
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib${LIBOMP_LIBDIR_SUFFIX})")
    endforeach()
else()
    install(TARGETS omp LIBRARY DESTINATION lib${LIBOMP_LIBDIR_SUFFIX}) 
    # Create aliases (symlinks) of the library for backwards compatibility
    set(LIBOMP_ALIASES "libgomp;libiomp5")
    foreach(alias IN LISTS LIBOMP_ALIASES)
        install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${lib_file}\" \"${alias}${dll}\"
            WORKING_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib${LIBOMP_LIBDIR_SUFFIX})")
    endforeach()
endif()
install(
    FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/omp.h 
    DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH}
)
if(${LIBOMP_OMPT_SUPPORT})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ompt.h DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH})
endif()

