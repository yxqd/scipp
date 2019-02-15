/// @file
/// SPDX-License-Identifier: GPL-3.0-or-later
/// @author Simon Heybrock
/// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
/// National Laboratory, and European Spallation Source ERIC.
#include <gtest/gtest.h>

#include <type_traits>

#include "test_macros.h"

#include "event_list_proxy.h"

TEST(ConstEventListProxy, length_mismatch_fail) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2};
  EXPECT_THROW_MSG(ConstEventListProxy x(a, b), std::runtime_error,
                   "Cannot zip data with mismatching length.");
}

TEST(EventListProxy, length_mismatch_fail) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2};
  EXPECT_THROW_MSG(EventListProxy x(a, b), std::runtime_error,
                   "Cannot zip data with mismatching length.");
}

TEST(ConstEventListProxy, from_vectors) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  ConstEventListProxy proxy(a, b);
  EXPECT_EQ(std::get<0>(*proxy.begin()), 1.1);
  EXPECT_EQ(std::get<1>(*proxy.begin()), 1);
}

TEST(EventListProxy, from_vectors) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  EventListProxy proxy(a, b);
  EXPECT_EQ(std::get<0>(*proxy.begin()), 1.1);
  EXPECT_EQ(std::get<1>(*proxy.begin()), 1);
  std::get<0>(*proxy.begin()) = 0.0;
  EXPECT_EQ(std::get<0>(*proxy.begin()), 0.0);
}

TEST(EventListProxy, push_back) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  EventListProxy proxy(a, b);
  proxy.push_back(4.4, 4);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  proxy.push_back(*proxy.begin());
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 4)), 1.1);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 4)), 1);
}

TEST(EventListProxy, push_back_3) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};
  std::vector<int32_t> c{3, 2, 1};
  EventListProxy proxy(a, b, c);
  proxy.push_back(4.4, 4, 1);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 3)), 1);
  proxy.push_back(*proxy.begin());
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 4)), 1.1);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 4)), 1);
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 4)), 3);
}

TEST(EventListProxy, push_back_duplicate_broken) {
  std::vector<double> a{1.1, 2.2, 3.3};
  std::vector<int32_t> b{1, 2, 3};

  // This is not allowed. We could add a check, but at this point it is not
  // clear if that is required, since creation should typically be under our
  // control, and we may want to avoid performance penalties.
  EventListProxy proxy(a, b, b);

  proxy.push_back(4.4, 4, 5);
  EXPECT_EQ(std::get<0>(*(proxy.begin() + 3)), 4.4);
  EXPECT_EQ(std::get<1>(*(proxy.begin() + 3)), 4);
  // b is now longer than a, we view the wrong element.
  EXPECT_EQ(std::get<2>(*(proxy.begin() + 3)), 4);
}

TEST(EventListsProxy, missing_field) {
  Dataset d;
  d.insert<double>(Data::Value, "a", {Dim::X, 4}, {1, 2, 3, 4});
  d.insert<float>(Data::Variance, "a", {Dim::X, 4}, {5, 6, 7, 8});

  EXPECT_THROW_MSG(
      EventListsProxy eventLists(d, Access::Key{Data::Value, "a"},
                                 Access::Key<float>{Data::Variance, "b"}),
      std::runtime_error,
      "Dataset with 2 variables, could not find variable with tag "
      "Data::Variance and name `b`.");
}

TEST(EventListsProxy, dimension_mismatch) {
  Dataset d;
  d.insert<double>(Data::Value, "a", {Dim::X, 4}, {1, 2, 3, 4});
  d.insert<float>(Data::Variance, "a", {}, {5});

  EXPECT_THROW_MSG(
      EventListsProxy eventLists(d, Access::Key{Data::Value, "a"},
                                 Access::Key<float>{Data::Variance, "a"}),
      std::runtime_error, "Event-data fields have mismatching dimensions.");
}

TEST(EventListsProxy, create) {
  Dataset d;
  d.insert<double>(Data::Value, "a", {Dim::X, 4}, {1, 2, 3, 4});
  d.insert<float>(Data::Variance, "a", {Dim::X, 4}, {5, 6, 7, 8});

  EventListsProxy eventLists(d, Access::Key{Data::Value, "a"},
                             Access::Key<float>{Data::Variance, "a"});
}
