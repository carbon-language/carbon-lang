.. _development_process:

Development Process Documentation
=================================

.. toctree::
   :hidden:

   CodingStandards
   MakefileGuide
   Projects

* :ref:`projects`

  How-to guide and templates for new projects that *use* the LLVM
  infrastructure.  The templates (directory organization, Makefiles, and test
  tree) allow the project code to be located outside (or inside) the ``llvm/``
  tree, while using LLVM header files and libraries.

* :ref:`coding_standards`

  Describes a few coding standards that are used in the LLVM source tree. All
  code submissions must follow the coding standards before being allowed into
  the source tree.

* `LLVMBuild Documentation <LLVMBuild.html>`_

  Describes the LLVMBuild organization and files used by LLVM to specify
  component descriptions.

* :ref:`makefile_guide`

  Describes how the LLVM makefiles work and how to use them.

* `How To Release LLVM To The Public <HowToReleaseLLVM.html>`_

  This is a guide to preparing LLVM releases. Most developers can ignore it.
