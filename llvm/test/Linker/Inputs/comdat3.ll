$foo = comdat noduplicates
@foo = global i64 43, comdat($foo)
