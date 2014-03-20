=====================
TableGen Deficiencies
=====================

.. contents::
   :local:

Introduction
============

Despite being very generic, TableGen has some deficiencies that have been
pointed out numerous times. The common theme is that, while TableGen allows
you to build Domain-Specific-Languages, the final languages that you create
lack the power of other DSLs, which in turn increase considerably the size
and complexity of TableGen files.

At the same time, TableGen allows you to create virtually any meaning of
the basic concepts via custom-made back-ends, which can pervert the original
design and make it very hard for newcomers to understand it.

There are some in favour of extending the semantics even more, but making sure
back-ends adhere to strict rules. Others suggesting we should move to more
powerful DSLs designed with specific purposes, or even re-using existing
DSLs.

Known Problems
==============

TODO: Add here frequently asked questions about why TableGen doesn't do
what you want, how it might, and how we could extend/restrict it to
be more use friendly.
