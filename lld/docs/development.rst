.. _development:

Development
===========

lld is developed as part of the `LLVM <http://llvm.org>`_ project.

Using C++11 in lld
------------------

:doc:`C++11`.

Creating a Reader
-----------------

See the :ref:`Creating a Reader <Readers>` guide.


Modifying the Driver
--------------------

See :doc:`Driver`.


Debugging
---------

You can run lld with ``-mllvm -debug`` command line options to enable debugging
printouts. If you want to enable debug information for some specific pass, you
can run it with ``-mllvm '-debug-only=<pass>'``, where pass is a name used in
the ``DEBUG_WITH_TYPE()`` macro.



Documentation
-------------

The project documentation is written in reStructuredText and generated using the
`Sphinx <http://sphinx.pocoo.org/>`_ documentation generator. For more
information on writing documentation for the project, see the
:ref:`sphinx_intro`.

.. toctree::
   :hidden:

   C++11
   Readers
   Driver
