.. title:: clang-tidy - abseil-duration-unnecessary-conversion

abseil-duration-unnecessary-conversion
======================================

Finds and fixes cases where ``absl::Duration`` values are being converted to
numeric types and back again.

Floating-point examples:

.. code-block:: c++

  // Original - Conversion to double and back again
  absl::Duration d1;
  absl::Duration d2 = absl::Seconds(absl::ToDoubleSeconds(d1));

  // Suggestion - Remove unnecessary conversions
  absl::Duration d2 = d1;

  // Original - Division to convert to double and back again
  absl::Duration d2 = absl::Seconds(absl::FDivDuration(d1, absl::Seconds(1)));

  // Suggestion - Remove division and conversion
  absl::Duration d2 = d1;

Integer examples:

.. code-block:: c++

  // Original - Conversion to integer and back again
  absl::Duration d1;
  absl::Duration d2 = absl::Hours(absl::ToInt64Hours(d1));

  // Suggestion - Remove unnecessary conversions
  absl::Duration d2 = d1;

  // Original - Integer division followed by conversion
  absl::Duration d2 = absl::Seconds(d1 / absl::Seconds(1));

  // Suggestion - Remove division and conversion
  absl::Duration d2 = d1;

Unwrapping scalar operations:

.. code-block:: c++

  // Original - Multiplication by a scalar
  absl::Duration d1;
  absl::Duration d2 = absl::Seconds(absl::ToInt64Seconds(d1) * 2);

  // Suggestion - Remove unnecessary conversion
  absl::Duration d2 = d1 * 2;

Note: Converting to an integer and back to an ``absl::Duration`` might be a
truncating operation if the value is not aligned to the scale of conversion.
In the rare case where this is the intended result, callers should use
``absl::Trunc`` to truncate explicitly.
