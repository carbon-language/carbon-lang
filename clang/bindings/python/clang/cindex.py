# -*- coding: utf-8 -*-

from ctypes import *

def get_cindex_library():
    # FIXME: It's probably not the case that the library is actually found in
    # this location. We need a better system of identifying and loading the
    # CIndex library. It could be on path or elsewhere, or versioned, etc.
    import platform
    name = platform.system()
    if name == 'Darwin':
        return cdll.LoadLibrary('libCIndex.dylib')
    elif name == 'Windows':
        return cdll.LoadLibrary('libCIndex.dll')
    else:
        return cdll.LoadLibrary('libCIndex.so')    
    
## Utility Types and Functions ##
def alloc_string_vector(strs):
    """
    Allocate a string buffer large enough to accommodate the given list of
    python strings.
    """
    n = 0
    for i in strs: n += len(i) + 1
    return create_string_buffer(n)

def copy_string_vector(vec, strs):
    """
    Copy the contents of each string into the vector, preserving null
    terminated elements.
    """
    n = 0
    for i in strs:
        # This is terribly inefficient, but I can't figure out how to copy a
        # chunk of characters into the resultant vector. t should be: something
        # like this: vec[n:n + len(i)] = i[:]; n += len(i) + 1
        for j in i:
            vec[n] = j
            n += 1
        n += 1

def create_string_vector(strs):
    """
    Create a string vector (char *[]) from the given list of strings.
    """
    vec = alloc_string_vector(strs)
    copy_string_vector(vec, strs)
    return vec

# Aliases for convenience
c_int_p = POINTER(c_int)
c_uint_p = POINTER(c_uint)
c_bool = c_uint

# ctypes doesn't implicitly convert c_void_p to the appropriate wrapper
# object. This is a problem, because it means that from_parameter will see an
# integer and pass the wrong value on platforms where int != void*. Work around
# this by marshalling object arguments as void**.
c_object_p = POINTER(c_void_p)

lib = get_cindex_library()

## Typedefs ##
CursorKind = c_int

### Structures and Utility Classes ###

class String(Structure):
    """
    The String class is a simple wrapper around constant string data returned
    from functions in the CIndex library.

    String objects do not provide any of the operations that Python strings
    support. However, these objects can be explicitly cast using the str()
    function.
    """
    _fields_ = [("spelling", c_char_p), ("free", c_int)]

    def __del__(self):
        if self.free:
            String_dispose(self)

    def __str__(self):
        return self.spelling

class SourceLocation(Structure):
    """
    A SourceLocation Represents a particular location within a source file.
    """
    _fields_ = [("ptr_data", c_void_p), ("int_data", c_uint)]

    def init(self):
        """
        Initialize the source location, setting its file, line and column.
        """
        f, l, c = c_void_p(), c_uint(), c_uint()
        SourceLocation_loc(self, byref(f), byref(l), byref(c))
        self.file, self.line, self.column = File(f), l, c
        return self

class SourceRange(Structure):
    """
    A SourceRange describes a range of source locations within the source
    code.
    """
    _fields_ = [
        ("ptr_data", c_void_p),
        ("begin_int_data", c_uint),
        ("end_int_data", c_uint)]

    def start(self):
        """
        Return a SourceLocation representing the first character within a
        source range.
        """
        return SourceRange_start(self).init()

    def end(self):
        """
        Return a SourceLocation representing the last character within a
        source range.
        """
        return SourceRange_end(self).init()

class Cursor(Structure):
    """
    The Cursor class represents a reference to an element within the AST. It
    acts as a kind of iterator.
    """
    _fields_ = [("kind", c_int), ("data", c_void_p * 3)]

    def __eq__(self, other):
        return Cursor_eq(self, other)

    def __ne__(self, other):
        return not Cursor_eq(self, other)

    @staticmethod
    def null():
        """Return the null cursor object."""
        return Cursor_null()

    def is_declaration(self):
        """Return True if the cursor points to a declaration."""
        return Cursor_is_decl(self.kind)

    def is_reference(self):
        """Return True if the cursor points to a refernce."""
        return Cursor_is_ref(self.kind)

    def is_expression(self):
        """Return True if the cursor points to an expression."""
        return Cursor_is_expr(self.kind)

    def is_statement(self):
        """Return True if the cursor points to a statement."""
        return Cursor_is_stmt(self.kind)

    def is_translation_unit(self):
        """Return True if the cursor points to a translation unit."""
        return Cursor_is_tu(self.kind)

    def is_invalid(self):
        """Return  True if the cursor points to an invalid entity."""
        return Cursor_is_inv(self.kind)

    def is_definition(self):
        """
        Returns true if the declaration pointed at by the cursor is also a
        definition of that entity.
        """
        return Cursor_is_def(self)

    def get_declaration(self):
        """
        Return the underlying declaration for the cursor. If the cursor kind
        is a declaration, then this simpy returns the declaration. If the
        cursor is a reference, then this returns the referenced declaration.
        """
        if not self.is_declaration():
            raise Exception("Cursor does not refer to a Declaration")
        return Cursor_decl(self)

    def get_definition(self):
        """
        If the cursor is a reference to a declaration or a declaration of
        some entity, return a cursor that points to the definition of that
        entity.
        """
        # TODO: Should probably check that this is either a reference or
        # declaration prior to issuing the lookup.
        return Cursor_def(self)

    @property
    def spelling(self):
        """Return the spelling of the entity pointed at by the cursor."""
        if not self.is_declaration():
            # FIXME: This should be documented in Index.h
            raise ValueError("Cursor does not refer to a Declaration")
        return Cursor_spelling(self)

    @property
    def location(self):
        """
        Return the source location (the starting character) of the entity
        pointed at by the cursor.
        """
        return Cursor_loc(self).init()

    @property
    def extent(self):
        """
        Return the source range (the range of text) occupied by the entity
        pointed at by the cursor.
        """
        return Cursor_extent(self)

    @property
    def file(self):
        """
        Return the file containing the pointed-at entity. This is an alias for
        location.file.
        """
        return self.location.file

# FIXME: Implement this class.
class Entity(Structure):
    """
    An Entity is a uniqe token for accessing "visible" declarations within
    a translation unit.
    """
    # NOTE: Index is written here as a void*, but given in the API as CXIndex.
    # Be careful to translate back to Index when returning this member.
    # TODO: Rename as _index and write a property?
    _fields_ = [("index", c_void_p), ("data", c_void_p)]

## CIndex Objects ##

# CIndex objects (derived from ClangObject) are essentially lightweight
# wrappers attached to some underlying object, which is exposed via CIndex as
# a void*.

class ClangObject(object):
    """
    A helper for Clang objects. This class helps act as an intermediary for
    the ctypes library and the Clang CIndex library.
    """
    def __init__(self, obj):
        assert isinstance(obj, c_object_p) and obj
        self.obj = self._as_parameter_ = obj

    def from_param(self):
        return self._as_parameter_

class Index(ClangObject):
    """
    The Index type provides the primary interface to the Clang CIndex library,
    primarily by providing an interface for reading and parsing translation
    units.
    """
    def __init__(self, obj):
        ClangObject.__init__(self, obj)

    @staticmethod
    def create(excludeDecls=False, displayDiags=False):
        """
        Create a new Index.
        Parameters:
        excludeDecls -- Exclude local declarations from translation units.
        displayDiags -- Display diagnostics during translation unit creation.
        """
        return Index(Index_create(excludeDecls, displayDiags))

    def __del__(self):
        Index_dispose(self)

    def read(self, path):
        """Load the translation unit from the given AST file."""
        return TranslationUnit.read(self, path)

    def parse(self, path, args = []):
        """
        Load the translation unit from the given source code file by running
        clang and generating the AST before loading. Additional command line
        parameters can be passed to clang via the args parameter.
        """
        return TranslationUnit.parse(self, path, args)


class TranslationUnit(ClangObject):
    """
    The TranslationUnit class represents a source code translation unit and
    provides read-only access to its top-level declarations.
    """
    def __init__(self, obj, free=False):
        ClangObject.__init__(self, obj)
        self.free = free

    def __del__(self):
        if self.free and self.obj:
            TranslationUnit_dispose(self)

    def load(self, fun, data = None):
        # Actually call this over a lambda that attaches an object the
        # underlying void pointer.
        f = lambda t, c, x: fun(TranslationUnit(t), c, x)
        TranslationUnit_load(self.obj, Callback(f), data)

    @property
    def spelling(self):
        return TranslationUnit_spelling(self)

    @staticmethod
    def read(ix, path):
        """Create a translation unit from the given AST file."""
        ptr = TranslationUnit_read(ix, path)
        return TranslationUnit(ptr, True) if ptr else None

    @staticmethod
    def parse(ix, path, args = []):
        """
        Construct a translation unit from the given source file, applying
        the given command line argument.
        """
        # TODO: Support unsaved files.
        argc, argv = len(args), create_string_vector(args)
        ptr = TranslationUnit_parse(ix, path, argc, byref(argv), 0, 0)
        return TranslationUnit(ptr, True) if ptr else None

class File(ClangObject):
    """
    The File class...
    """
    def __init__(self, obj):
        ClangObject.__init__(self, obj)

    @property
    def is_valid(self):
        return self.obj is not None

    @property
    def name(self):
        """Return the name of the file, if valid. Otherwise, an empty string."""
        return File_name(self) if self.obj else ""

    @property
    def time(self):
        """Return the time of the file, if valid. Otherwise, -1."""
        return File_time(self) if self.obj else -1

class Declaration(ClangObject):
    """
    The Declaration class represents a declaration with a translation unit.
    """
    def __init__(self, obj):
        ClangObject.__init__(self, obj)

        # Figure out the kind of cursor and inject a base class that provides
        # some declaration-specific functionality.
        self.cursor = Declaration_cursor(self)

    @property
    def kind(self):
        """Retur the kind of cursor."""
        return self.cursor.kind

    @property
    def entity(self):
        """Return an entity that represents this declaration."""
        return Entity(Declaration_entity(self))

    @property
    def spelling(self):
        """Return the spelling (name) of the declaration."""
        return Declaration_spelling(self)

    def load(self, fun, data = None):
        """
        Recursively visit any elements declared or referenced within this
        declaration.
        """
        f = lambda d, c, x: fun(Declaration(d), c, x)
        Declaration_load(self, Callback(f), data)

# Specific declaration kinds
class ClassDeclaration:
    pass

class FunctionDeclaration:
    pass

class TypedefDeclaration:
    pass

# Additional Functions and Types

# Wrap calls to TranslationUnit._load and Decl._load.
Callback = CFUNCTYPE(None, c_void_p, Cursor, c_void_p)

# String Functions
String_dispose = lib.clang_disposeString
String_dispose.argtypes = [String]

# Source Location Functions
SourceLocation_loc = lib.clang_getInstantiationLocation
SourceLocation_loc.argtypes = [SourceLocation, POINTER(c_void_p), c_uint_p, c_uint_p]

# Source Range Functions
SourceRange_start = lib.clang_getRangeStart
SourceRange_start.argtypes = [SourceLocation]
SourceRange_start.restype = SourceRange

SourceRange_end = lib.clang_getRangeEnd
SourceRange_end.argtypes = [SourceLocation]
SourceRange_end.restype = SourceRange

# Cursor Functions
# TODO: Implement this function
Cursor_get = lib.clang_getCursor
Cursor_get.argtypes = [TranslationUnit, c_char_p, c_uint, c_uint]
Cursor.restype = Cursor

Cursor_null = lib.clang_getNullCursor
Cursor_null.restype = Cursor

Cursor_kind = lib.clang_getCursorKind
Cursor_kind.argtypes = [Cursor]
Cursor_kind.res = c_int

# FIXME: Not really sure what a USR is or what this function actually does...
Cursor_usr = lib.clang_getCursorUSR

Cursor_is_decl = lib.clang_isDeclaration
Cursor_is_decl.argtypes = [CursorKind]
Cursor_is_decl.restype = c_bool

Cursor_is_ref = lib.clang_isReference
Cursor_is_ref.argtypes = [CursorKind]
Cursor_is_ref.restype = c_bool

Cursor_is_expr = lib.clang_isExpression
Cursor_is_expr.argtypes = [CursorKind]
Cursor_is_expr.restype = c_bool

Cursor_is_stmt = lib.clang_isStatement
Cursor_is_stmt.argtypes = [CursorKind]
Cursor_is_stmt.restype = c_bool

Cursor_is_inv = lib.clang_isInvalid
Cursor_is_inv.argtypes = [CursorKind]
Cursor_is_inv.restype = c_bool

Cursor_is_tu = lib.clang_isTranslationUnit
Cursor_is_tu.argtypes = [CursorKind]
Cursor_is_tu.restype = c_bool

Cursor_is_def = lib.clang_isCursorDefinition
Cursor_is_def.argtypes = [Cursor]
Cursor_is_def.restype = c_bool

Cursor_def = lib.clang_getCursorDefinition
Cursor_def.argtypes = [Cursor]
Cursor_def.restype = Cursor

Cursor_eq = lib.clang_equalCursors
Cursor_eq.argtypes = [Cursor, Cursor]
Cursor_eq.restype = c_uint

Cursor_spelling = lib.clang_getCursorSpelling
Cursor_spelling.argtypes = [Cursor]
Cursor_spelling.restype = String

Cursor_loc = lib.clang_getCursorLocation
Cursor_loc.argtypes = [Cursor]
Cursor_loc.restype = SourceLocation

Cursor_extent = lib.clang_getCursorExtent
Cursor_extent.argtypes = [Cursor]
Cursor_extent.restype = SourceRange

Cursor_ref = lib.clang_getCursorReferenced
Cursor_ref.argtypes = [Cursor]
Cursor_ref.restype = Cursor

# Index Functions
Index_create = lib.clang_createIndex
Index_create.argtypes = [c_int, c_int]
Index_create.restype = c_object_p

Index_dispose = lib.clang_disposeIndex
Index_dispose.argtypes = [Index]

# Translation Unit Functions
TranslationUnit_read = lib.clang_createTranslationUnit
TranslationUnit_read.argtypes = [Index, c_char_p]
TranslationUnit_read.restype = c_object_p

TranslationUnit_parse = lib.clang_createTranslationUnitFromSourceFile
TranslationUnit_parse.argtypes = [Index, c_char_p, c_int, c_void_p,
                                  c_int, c_void_p]
TranslationUnit_parse.restype = c_object_p

TranslationUnit_spelling = lib.clang_getTranslationUnitSpelling
TranslationUnit_spelling.argtypes = [TranslationUnit]
TranslationUnit_spelling.restype = String

TranslationUnit_dispose = lib.clang_disposeTranslationUnit
TranslationUnit_dispose.argtypes = [TranslationUnit]

# File Functions
File_name = lib.clang_getFileName
File_name.argtypes = [File]
File_name.restype = c_char_p

File_time = lib.clang_getFileTime
File_time.argtypes = [File]
File_time.restype = c_uint
