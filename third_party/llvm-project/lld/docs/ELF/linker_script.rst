Linker Script implementation notes and policy
=============================================

LLD implements a large subset of the GNU ld linker script notation. The LLD
implementation policy is to implement linker script features as they are
documented in the ld `manual <https://sourceware.org/binutils/docs/ld/Scripts.html>`_
We consider it a bug if the lld implementation does not agree with the manual
and it is not mentioned in the exceptions below.

The ld manual is not a complete specification, and is not sufficient to build
an implementation. In particular some features are only defined by the
implementation and have changed over time.

The lld implementation policy for properties of linker scripts that are not
defined by the documentation is to follow the GNU ld implementation wherever
possible. We reserve the right to make different implementation choices where
it is appropriate for LLD. Intentional deviations will be documented in this
file.

Symbol assignment
~~~~~~~~~~~~~~~~~

A symbol assignment looks like:

::

  symbol = expression;
  symbol += expression;

The first form defines ``symbol``. If ``symbol`` is already defined, it will be
overridden. The other form requires ``symbol`` to be already defined.

For a simple assignment like ``alias = aliasee;``, the ``st_type`` field is
copied from the original symbol. Any arithmetic operation (e.g. ``+ 0`` will
reset ``st_type`` to ``STT_NOTYPE``.

The ``st_size`` field is set to 0.

SECTIONS command
~~~~~~~~~~~~~~~~

A ``SECTIONS`` command looks like:

::

  SECTIONS {
    section-command
    section-command
    ...
  } [INSERT [AFTER|BEFORE] anchor_section;]

Each section-command can be a symbol assignment, an output section description,
or an overlay description.

When the ``INSERT`` keyword is present, the ``SECTIONS`` command describes some
output sections which should be inserted after or before the specified anchor
section. The insertion occurs after input sections have been mapped to output
sections but before orphan sections have been processed.

In the case where no linker script has been provided or every ``SECTIONS``
command is followed by ``INSERT``, LLD applies built-in rules which are similar
to GNU ld's internal linker scripts.

- Align the first section in a ``PT_LOAD`` segment according to ``-z noseparate-code``,
  ``-z separate-code``, or ``-z separate-loadable-segments``
- Define ``__bss_start``, ``end``, ``_end``, ``etext``, ``_etext``, ``edata``, ``_edata``
- Sort ``.ctors.*``/``.dtors.*``/``.init_array.*``/``.fini_array.*`` and PowerPC64 specific ``.toc``
- Place input ``.text.*`` into output ``.text``, and handle certain variants
  (``.text.hot.``, ``.text.unknown.``, ``.text.unlikely.``, etc) in the precense of
  ``-z keep-text-section-prefix``.

Output section description
~~~~~~~~~~~~~~~~~~~~~~~~~~

The description of an output section looks like:

::

  section [address] [(type)] : [AT(lma)] [ALIGN(section_align)] [SUBALIGN](subsection_align)] {
    output-section-command
    ...
  } [>region] [AT>lma_region] [:phdr ...] [=fillexp] [,]

Output section address
----------------------

When an *OutputSection* *S* has ``address``, LLD will set sh_addr to ``address``.

The ELF specification says:

> The value of sh_addr must be congruent to 0, modulo the value of sh_addralign.

The presence of ``address`` can cause the condition unsatisfied. LLD will warn.
GNU ld from Binutils 2.35 onwards will reduce sh_addralign so that
sh_addr=0 (modulo sh_addralign).

Output section type
-------------------

When an *OutputSection* *S* has ``(type)``, LLD will set ``sh_type`` or
``sh_flags`` of *S*. ``type`` is one of:

- ``NOLOAD``: set ``sh_type`` to ``SHT_NOBITS``.
- ``COPY``, ``INFO``, ``OVERLAY``: clear the ``SHF_ALLOC`` bit in ``sh_flags``.
- ``TYPE=<value>``: set ``sh_type`` to the specified value. ``<value>`` must be
  an integer or one of ``SHT_PROGBITS, SHT_NOTE, SHT_NOBITS, SHT_INIT_ARRAY,
  SHT_FINI_ARRAY, SHT_PREINIT_ARRAY``.

When ``sh_type`` is specified, it is an error if an input section in *S* has a
different type.

Output section alignment
------------------------

sh_addralign of an *OutputSection* *S* is the maximum of
``ALIGN(section_align)`` and the maximum alignment of the input sections in
*S*.

When an *OutputSection* *S* has both ``address`` and ``ALIGN(section_align)``,
GNU ld will set sh_addralign to ``ALIGN(section_align)``.

Output section LMA
------------------

A load address (LMA) can be specified by ``AT(lma)`` or ``AT>lma_region``.

- ``AT(lma)`` specifies the exact load address. If the linker script does not
  have a PHDRS command, then a new loadable segment will be generated.
- ``AT>lma_region`` specifies the LMA region. The lack of ``AT>lma_region``
  means the default region is used. Note, GNU ld propagates the previous LMA
  memory region when ``address`` is not specified. The LMA is set to the
  current location of the memory region aligned to the section alignment.
  If the linker script does not have a PHDRS command, then if
  ``lma_region`` is different from the ``lma_region`` for
  the previous OutputSection a new loadable segment will be generated.

The two keywords cannot be specified at the same time.

If neither ``AT(lma)`` nor ``AT>lma_region`` is specified:

- If the previous section is also in the default LMA region, and the two
  section have the same memory regions, the difference between the LMA and the
  VMA is computed to be the same as the previous difference.
- Otherwise, the LMA is set to the VMA.

Overwrite sections
~~~~~~~~~~~~~~~~~~

An ``OVERWRITE_SECTIONS`` command looks like:

::

  OVERWRITE_SECTIONS {
    output-section-description
    output-section-description
    ...
  }

Unlike a ``SECTIONS`` command, ``OVERWRITE_SECTIONS``  does not specify a
section order or suppress the built-in rules.

If a described output section description also appears in a ``SECTIONS``
command, the ``OVERWRITE_SECTIONS`` command wins; otherwise, the output section
will be added somewhere following the usual orphan section placement rules.

If a described output section description also appears in an ``INSERT
[AFTER|BEFORE]`` command, the description will be provided by the
description in the ``OVERWRITE_SECTIONS`` command while the insert command
still applies (possibly after orphan section placement). It is recommended to
leave the brace empty (i.e. ``section : {}``) for the insert command, because
its description will be ignored anyway.
