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
#include <thrust/detail/tuple/variadic_tuple.h>
#include <thrust/pair.h>
#include <thrust/detail/reference_forward_declaration.h>

namespace thrust
{
namespace detail
{

  
template<typename... Types>
  class tuple_of_iterator_references
    : public thrust::tuple<Types...>
{
  private:
    typedef thrust::tuple<Types...> super_t;

  public:
    // XXX we shouldn't need this, it's taken into account with the using super_t::super_t below
    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t &other)
      : super_t(other)
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    template<typename... UTypes>
    inline __host__ __device__
    tuple_of_iterator_references& operator=(const thrust::tuple<UTypes...> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from pairs
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const thrust::pair<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    // XXX perhaps we should generalize to reference<T>
    //     we could captures reference<pair> this way
    template<typename... UTypes,
             typename Pointer, typename Derived>
    inline __host__ __device__
// XXX gcc-4.2 crashes on is_assignable
//    typename thrust::detail::enable_if<
//      thrust::detail::is_assignable<
//        super_t,
//        const thrust::tuple<UTypes...>
//      >::value,
//      tuple_of_iterator_references &
//    >::type
    tuple_of_iterator_references &
    operator=(const thrust::reference<thrust::tuple<UTypes...>, Pointer, Derived> &other)
    {
      typedef thrust::tuple<UTypes...> tuple_type;

      // XXX perhaps this could be accelerated
      tuple_type other_tuple = other;
      super_t::operator=(other_tuple);
      return *this;
    }


    // duplicate thrust::tuple's constructors
    using super_t::super_t;
};


template<typename T> struct is_tuple_of_iterator_references : thrust::detail::false_type {};

template<typename... Types>
  struct is_tuple_of_iterator_references<
    thrust::detail::tuple_of_iterator_references<
      Types...
    >
  >
    : thrust::detail::true_type
{};


} // end detail


// specialize tuple_size for tuple_of_iterator_references
template<typename... Types>
struct tuple_size<detail::tuple_of_iterator_references<Types...>> : thrust::tuple_size<thrust::tuple<Types...>> {};

// specialize tuple_element for tuple_of_iterator_references
template<int i, typename... Types>
struct tuple_element<i,detail::tuple_of_iterator_references<Types...>> : thrust::tuple_element<i,thrust::tuple<Types...>> {};


} // end thrust


