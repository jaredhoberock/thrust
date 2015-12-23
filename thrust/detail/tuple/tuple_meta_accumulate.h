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
#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{


// define apply2 for tuple_meta_accumulate_impl
template<typename UnaryMetaFunctionClass, class Arg1, class Arg2>
  struct apply2
    : UnaryMetaFunctionClass::template apply<Arg1,Arg2>
{
}; // end apply2


template<int i, int end, typename Tuple, class BinaryMetaFunction, class StartType>
struct tuple_meta_accumulate_impl
{
  typedef typename apply2<
    BinaryMetaFunction,
    typename thrust::tuple_element<i,Tuple>::type,
    typename tuple_meta_accumulate_impl<i+1,end,Tuple,BinaryMetaFunction,StartType>::type
  >::type type;
};

template<int i, typename Tuple, class BinaryMetaFunction, class StartType>
struct tuple_meta_accumulate_impl<i,i,Tuple,BinaryMetaFunction,StartType>
{
  typedef StartType type;
};

// Meta-accumulate algorithm for tuples. Note: The template 
// parameter StartType corresponds to the initial value in 
// ordinary accumulation.
//
template<typename Tuple, class BinaryMetaFunction, class StartType>
struct tuple_meta_accumulate
  : tuple_meta_accumulate_impl<0,thrust::tuple_size<Tuple>::value,Tuple,BinaryMetaFunction,StartType> 
{};


} // end detail
} // end thrust

