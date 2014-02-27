include(CheckCXXCompilerFlag)
include(CheckSymbolExists)

# CodeGen options.
check_cxx_compiler_flag(-fPIC                COMPILER_RT_HAS_FPIC_FLAG)
check_cxx_compiler_flag(-fPIE                COMPILER_RT_HAS_FPIE_FLAG)
check_cxx_compiler_flag(-fno-builtin         COMPILER_RT_HAS_FNO_BUILTIN_FLAG)
check_cxx_compiler_flag(-fno-exceptions      COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fomit-frame-pointer COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG)
check_cxx_compiler_flag(-funwind-tables      COMPILER_RT_HAS_FUNWIND_TABLES_FLAG)
check_cxx_compiler_flag(-fno-stack-protector COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG)
check_cxx_compiler_flag(-fvisibility=hidden  COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG)
check_cxx_compiler_flag(-fno-rtti            COMPILER_RT_HAS_FNO_RTTI_FLAG)
check_cxx_compiler_flag(-ffreestanding       COMPILER_RT_HAS_FFREESTANDING_FLAG)
check_cxx_compiler_flag("-Werror -fno-function-sections" COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG)

check_cxx_compiler_flag(/GR COMPILER_RT_HAS_GR_FLAG)
check_cxx_compiler_flag(/GS COMPILER_RT_HAS_GS_FLAG)
check_cxx_compiler_flag(/MT COMPILER_RT_HAS_MT_FLAG)
check_cxx_compiler_flag(/Oy COMPILER_RT_HAS_Oy_FLAG)

# Debug info flags.
check_cxx_compiler_flag(-gline-tables-only COMPILER_RT_HAS_GLINE_TABLES_ONLY_FLAG)
check_cxx_compiler_flag(-g COMPILER_RT_HAS_G_FLAG)
check_cxx_compiler_flag(/Zi COMPILER_RT_HAS_Zi_FLAG)
 
# Warnings.
check_cxx_compiler_flag(-Werror COMPILER_RT_HAS_WERROR_FLAG)
check_cxx_compiler_flag("-Werror -Wframe-larger-than=512" COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG)
check_cxx_compiler_flag("-Werror -Wglobal-constructors"   COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG)
check_cxx_compiler_flag("-Werror -Wno-c99-extensions"     COMPILER_RT_HAS_WNO_C99_EXTENSIONS_FLAG)
check_cxx_compiler_flag("-Werror -Wno-gnu"                COMPILER_RT_HAS_WNO_GNU_FLAG)
check_cxx_compiler_flag("-Werror -Wno-non-virtual-dtor"   COMPILER_RT_HAS_WNO_NON_VIRTUAL_DTOR_FLAG)
check_cxx_compiler_flag("-Werror -Wno-variadic-macros"    COMPILER_RT_HAS_WNO_VARIADIC_MACROS_FLAG)

check_cxx_compiler_flag(/WX COMPILER_RT_HAS_WX_FLAG)
check_cxx_compiler_flag(/wd4722 COMPILER_RT_HAS_WD4722_FLAG)

# Symbols.
check_symbol_exists(__func__ "" COMPILER_RT_HAS_FUNC_SYMBOL)
