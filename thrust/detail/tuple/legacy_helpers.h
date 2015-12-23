/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

// these helpers were introduced by the pre c++11 implementation of thrust::tuple and are retained here

namespace thrust
{

// define null_type
struct null_type {};

// null_type comparisons
__host__ __device__ inline
bool operator==(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator<(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator>(const null_type&, const null_type&) { return false; }



template <class T> struct access_traits
{
  typedef const T& const_type;
  typedef T& non_const_type;

  typedef const typename thrust::detail::remove_cv<T>::type& parameter_type;

// used as the tuple constructors parameter types
// Rationale: non-reference tuple element types can be cv-qualified.
// It should be possible to initialize such types with temporaries,
// and when binding temporaries to references, the reference must
// be non-volatile and const. 8.5.3. (5)
}; // end access_traits

template <class T> struct access_traits<T&>
{
  typedef T& const_type;
  typedef T& non_const_type;

  typedef T& parameter_type;
}; // end access_traits<T&>


} // end thrust

