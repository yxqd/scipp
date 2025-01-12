// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_PYTHON_BIND_OPERATORS_H
#define SCIPP_PYTHON_BIND_OPERATORS_H

#include "pybind11.h"

namespace py = pybind11;

template <class Other, class T, class... Ignored>
void bind_comparison(pybind11::class_<T, Ignored...> &c) {
  c.def("__eq__", [](T &a, Other &b) { return a == b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__ne__", [](T &a, Other &b) { return a != b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_in_place_binary(pybind11::class_<T, Ignored...> &c) {
  c.def("__iadd__", [](T &a, Other &b) { return a += b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__isub__", [](T &a, Other &b) { return a -= b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__imul__", [](T &a, Other &b) { return a *= b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__itruediv__", [](T &a, Other &b) { return a /= b; },
        py::is_operator(), py::call_guard<py::gil_scoped_release>());
}

template <class Other, class T, class... Ignored>
void bind_binary(pybind11::class_<T, Ignored...> &c) {
  c.def("__add__", [](T &a, Other &b) { return a + b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__sub__", [](T &a, Other &b) { return a - b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__mul__", [](T &a, Other &b) { return a * b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
  c.def("__truediv__", [](T &a, Other &b) { return a / b; }, py::is_operator(),
        py::call_guard<py::gil_scoped_release>());
}

#endif // SCIPP_PYTHON_BIND_OPERATORS_H
