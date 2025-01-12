// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_UNITS_EXCEPT_H
#define SCIPP_UNITS_EXCEPT_H

#include "scipp-units_export.h"
#include "scipp/common/except.h"
#include "scipp/units/unit.h"

namespace scipp::units {

SCIPP_UNITS_EXPORT std::string to_string(const Unit &unit);
}

namespace scipp::except {

using UnitError = Error<units::Unit>;
using UnitMismatchError = MismatchError<units::Unit>;

// We need deduction guides such that, e.g., the exception for a Variable
// mismatch and VariableProxy mismatch are the same type.
template <class T>
MismatchError(const units::Unit &, const T &)->MismatchError<units::Unit>;

} // namespace scipp::except

#endif // SCIPP_UNITS_EXCEPT_H
