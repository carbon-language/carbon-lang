===============================================
Static Analyzer Design Document: Memory Regions
===============================================

Authors: Ted Kremenek, ``kremenek at apple``,
Zhongxing Xu, ``xuzhongzhing at gmail``

Introduction
============

The path-sensitive analysis engine in libAnalysis employs an extensible
API for abstractly modeling the memory of an analyzed program. This API
employs the concept of "memory regions" to abstractly model chunks of
program memory such as program variables and dynamically allocated
memory such as those returned from 'malloc' and 'alloca'. Regions are
hierarchical, with subregions modeling subtyping relationships, field
and array offsets into larger chunks of memory, and so on.

The region API consists of two components:

-  A taxonomy and representation of regions themselves within the
   analyzer engine. The primary definitions and interfaces are described
   in ``MemRegion.h``. At the root of the region hierarchy is the class
   ``MemRegion`` with specific subclasses refining the region concept
   for variables, heap allocated memory, and so forth.
-  The modeling of binding of values to regions. For example, modeling
   the value stored to a local variable ``x`` consists of recording the
   binding between the region for ``x`` (which represents the raw memory
   associated with ``x``) and the value stored to ``x``. This binding
   relationship is captured with the notion of "symbolic stores."

Symbolic stores, which can be thought of as representing the relation
``regions -> values``, are implemented by subclasses of the
``StoreManager`` class (``Store.h``). A particular StoreManager
implementation has complete flexibility concerning the following:

-  *How* to model the binding between regions and values
-  *What* bindings are recorded

Together, both points allow different StoreManagers to tradeoff between
different levels of analysis precision and scalability concerning the
reasoning of program memory. Meanwhile, the core path-sensitive engine
makes no assumptions about either points, and queries a StoreManager
about the bindings to a memory region through a generic interface that
all StoreManagers share. If a particular StoreManager cannot reason
about the potential bindings of a given memory region (e.g.,
'``BasicStoreManager``' does not reason about fields of structures) then
the StoreManager can simply return 'unknown' (represented by
'``UnknownVal``') for a particular region-binding. This separation of
concerns not only isolates the core analysis engine from the details of
reasoning about program memory but also facilities the option of a
client of the path-sensitive engine to easily swap in different
StoreManager implementations that internally reason about program memory
in very different ways.

The rest of this document is divided into two parts. We first discuss
region taxonomy and the semantics of regions. We then discuss the
StoreManager interface, and details of how the currently available
StoreManager classes implement region bindings.

Memory Regions and Region Taxonomy
==================================

Pointers
--------

Before talking about the memory regions, we would talk about the
pointers since memory regions are essentially used to represent pointer
values.

The pointer is a type of values. Pointer values have two semantic
aspects. One is its physical value, which is an address or location. The
other is the type of the memory object residing in the address.

Memory regions are designed to abstract these two properties of the
pointer. The physical value of a pointer is represented by MemRegion
pointers. The rvalue type of the region corresponds to the type of the
pointee object.

One complication is that we could have different view regions on the
same memory chunk. They represent the same memory location, but have
different abstract location, i.e., MemRegion pointers. Thus we need to
canonicalize the abstract locations to get a unique abstract location
for one physical location.

Furthermore, these different view regions may or may not represent
memory objects of different types. Some different types are semantically
the same, for example, 'struct s' and 'my\_type' are the same type.

::

    struct s;
    typedef struct s my_type;

But ``char`` and ``int`` are not the same type in the code below:

::

    void *p;
    int *q = (int*) p;
    char *r = (char*) p;

Thus we need to canonicalize the MemRegion which is used in binding and
retrieving.

Regions
-------

Region is the entity used to model pointer values. A Region has the
following properties:

-  Kind
-  ObjectType: the type of the object residing on the region.
-  LocationType: the type of the pointer value that the region
   corresponds to. Usually this is the pointer to the ObjectType. But
   sometimes we want to cache this type explicitly, for example, for a
   CodeTextRegion.
-  StartLocation
-  EndLocation

Symbolic Regions
----------------

A symbolic region is a map of the concept of symbolic values into the
domain of regions. It is the way that we represent symbolic pointers.
Whenever a symbolic pointer value is needed, a symbolic region is
created to represent it.

A symbolic region has no type. It wraps a SymbolData. But sometimes we
have type information associated with a symbolic region. For this case,
a TypedViewRegion is created to layer the type information on top of the
symbolic region. The reason we do not carry type information with the
symbolic region is that the symbolic regions can have no type. To be
consistent, we don't let them to carry type information.

Like a symbolic pointer, a symbolic region may be NULL, has unknown
extent, and represents a generic chunk of memory.

.. note::
   We plan not to use loc::SymbolVal in RegionStore and remove it
   gradually.

Symbolic regions get their rvalue types through the following ways:

-  Through the parameter or global variable that points to it, e.g.:

   ::

       void f(struct s* p) {
         ...
       }

   The symbolic region pointed to by ``p`` has type ``struct s``.

-  Through explicit or implicit casts, e.g.:

   ::

       void f(void* p) {
         struct s* q = (struct s*) p;
         ...
       }

We attach the type information to the symbolic region lazily. For the
first case above, we create the ``TypedViewRegion`` only when the
pointer is actually used to access the pointee memory object, that is
when the element or field region is created. For the cast case, the
``TypedViewRegion`` is created when visiting the ``CastExpr``.

The reason for doing lazy typing is that symbolic regions are sometimes
only used to do location comparison.

Pointer Casts
-------------

Pointer casts allow people to impose different 'views' onto a chunk of
memory.

Usually we have two kinds of casts. One kind of casts cast down with in
the type hierarchy. It imposes more specific views onto more generic
memory regions. The other kind of casts cast up with in the type
hierarchy. It strips away more specific views on top of the more generic
memory regions.

We simulate the down casts by layering another ``TypedViewRegion`` on
top of the original region. We simulate the up casts by striping away
the top ``TypedViewRegion``. Down casts is usually simple. For up casts,
if the there is no ``TypedViewRegion`` to be stripped, we return the
original region. If the underlying region is of the different type than
the cast-to type, we flag an error state.

For toll-free bridging casts, we return the original region.

We can set up a partial order for pointer types, with the most general
type ``void*`` at the top. The partial order forms a tree with ``void*``
as its root node.

Every ``MemRegion`` has a root position in the type tree. For example,
the pointee region of ``void *p`` has its root position at the root node
of the tree. ``VarRegion`` of ``int x`` has its root position at the
'int type' node.

``TypedViewRegion`` is used to move the region down or up in the tree.
Moving down in the tree adds a ``TypedViewRegion``. Moving up in the
tree removes a ``TypedViewRegion``.

Do we want to allow moving up beyond the root position? This happens
when:

::

     int x; void *p = &x; 

The region of ``x`` has its root position at 'int\*' node. the cast to
void\* moves that region up to the 'void\*' node. I propose to not allow
such casts, and assign the region of ``x`` for ``p``.

Another non-ideal case is that people might cast to a non-generic
pointer from another non-generic pointer instead of first casting it
back to the generic pointer. Direct handling of this case would result
in multiple layers of TypedViewRegions. This enforces an incorrect
semantic view to the region, because we can only have one typed view on
a region at a time. To avoid this inconsistency, before casting the
region, we strip the TypedViewRegion, then do the cast. In summary, we
only allow one layer of TypedViewRegion.

Region Bindings
---------------

The following region kinds are boundable: VarRegion,
CompoundLiteralRegion, StringRegion, ElementRegion, FieldRegion, and
ObjCIvarRegion.

When binding regions, we perform canonicalization on element regions and
field regions. This is because we can have different views on the same
region, some of which are essentially the same view with different sugar
type names.

To canonicalize a region, we get the canonical types for all
TypedViewRegions along the way up to the root region, and make new
TypedViewRegions with those canonical types.

For Objective-C and C++, perhaps another canonicalization rule should be
added: for FieldRegion, the least derived class that has the field is
used as the type of the super region of the FieldRegion.

All bindings and retrievings are done on the canonicalized regions.

Canonicalization is transparent outside the region store manager, and
more specifically, unaware outside the Bind() and Retrieve() method. We
don't need to consider region canonicalization when doing pointer cast.

Constraint Manager
------------------

The constraint manager reasons about the abstract location of memory
objects. We can have different views on a region, but none of these
views changes the location of that object. Thus we should get the same
abstract location for those regions.
