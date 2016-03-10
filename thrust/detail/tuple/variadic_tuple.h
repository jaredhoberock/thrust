// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <thrust/detail/config.h>
#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <thrust/pair.h>
#include <thrust/detail/tuple/legacy_helpers.h>
#include <thrust/detail/swap.h>


// let the rest of the library know that thrust::tuple is variadic
#define __thrust_lib_has_variadic_tuple 1


namespace thrust
{

template<class... Types> class tuple;


template<int, class>
struct tuple_element;


template<int i>
struct tuple_element<i, thrust::tuple<>> {};


template<class Type1, class... Types>
struct tuple_element<0, thrust::tuple<Type1,Types...>>
{
  using type = Type1;
};


template<int i, class Type1, class... Types>
struct tuple_element<i, thrust::tuple<Type1,Types...>>
{
  using type = typename tuple_element<i - 1, thrust::tuple<Types...>>::type;
};


template<class>
struct tuple_size;


template<class... Types>
struct tuple_size<thrust::tuple<Types...>>
  : std::integral_constant<size_t, sizeof...(Types)>
{};


namespace detail
{

// define variadic "and" operator 
template <typename... Conditions>
  struct tuple_and;

template<>
  struct tuple_and<>
    : public std::true_type
{
};

template <typename Condition, typename... Conditions>
  struct tuple_and<Condition, Conditions...>
    : public std::integral_constant<
        bool,
        Condition::value && tuple_and<Conditions...>::value>
{
};

// XXX this implementation is based on Howard Hinnant's "tuple leaf" construction in libcxx


// define index sequence in case it is missing
// prefix this stuff with "tuple" to avoid collisions with other implementations
template<size_t... I> struct tuple_index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct tuple_make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct tuple_make_index_sequence_impl<
  Start,
  tuple_index_sequence<Indices...>, 
  End
>
{
  typedef typename tuple_make_index_sequence_impl<
    Start + 1,
    tuple_index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct tuple_make_index_sequence_impl<End, tuple_index_sequence<Indices...>, End>
{
  typedef tuple_index_sequence<Indices...> type;
};

template<size_t N>
using tuple_make_index_sequence = typename tuple_make_index_sequence_impl<0, tuple_index_sequence<>, N>::type;


template<class T>
struct tuple_use_empty_base_class_optimization
  : std::integral_constant<
      bool,
      std::is_empty<T>::value
#if __cplusplus >= 201402L
      && !std::is_final<T>::value
#endif
    >
{};


template<class T, bool = tuple_use_empty_base_class_optimization<T>::value>
class tuple_leaf_base
{
  public:
    __thrust_exec_check_disable__
    __host__ __device__
    tuple_leaf_base() = default;

    __thrust_exec_check_disable__
    template<class U>
    __host__ __device__
    tuple_leaf_base(U&& arg) : val_(std::forward<U>(arg)) {}

    __host__ __device__
    const T& const_get() const
    {
      return val_;
    }

    __host__ __device__
    T& mutable_get()
    {
      return val_;
    }

  private:
    T val_;
};

template<class T>
class tuple_leaf_base<T,true> : public T
{
  public:
    __host__ __device__
    tuple_leaf_base() = default;

    template<class U>
    __host__ __device__
    tuple_leaf_base(U&& arg) : T(std::forward<U>(arg)) {}

    __host__ __device__
    const T& const_get() const
    {
      return *this;
    }
  
    __host__ __device__
    T& mutable_get()
    {
      return *this;
    }
};

template<size_t I, class T>
class tuple_leaf : public tuple_leaf_base<T>
{
  private:
    using super_t = tuple_leaf_base<T>;

  public:
    __host__ __device__
    tuple_leaf() = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __host__ __device__
    tuple_leaf(U&& arg) : super_t(std::forward<U>(arg)) {}

    __host__ __device__
    tuple_leaf(const tuple_leaf& other) : super_t(other.const_get()) {}

    __host__ __device__
    tuple_leaf(tuple_leaf&& other) : super_t(std::forward<T>(other.mutable_get())) {}

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,const U&>::value
             >::type>
    __host__ __device__
    tuple_leaf(const tuple_leaf<I,U>& other) : super_t(other.const_get()) {}

    // converting move-constructor
    // note the use of std::forward<U> here to allow construction of T from U&&
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U&&>::value
             >::type>
    __host__ __device__
    tuple_leaf(tuple_leaf<I,U>&& other) : super_t(std::forward<U>(other.mutable_get())) {}


    __thrust_exec_check_disable__
    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U>::value
             >::type>
    __host__ __device__
    tuple_leaf& operator=(const tuple_leaf<I,U>& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }
    
    __thrust_exec_check_disable__
    __host__ __device__
    tuple_leaf& operator=(const tuple_leaf& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }

    __thrust_exec_check_disable__
    __host__ __device__
    tuple_leaf& operator=(tuple_leaf&& other)
    {
      this->mutable_get() = std::forward<T>(other.mutable_get());
      return *this;
    }

    __thrust_exec_check_disable__
    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U&&>::value
             >::type>
    __host__ __device__
    tuple_leaf& operator=(tuple_leaf<I,U>&& other)
    {
      this->mutable_get() = std::forward<U>(other.mutable_get());
      return *this;
    }

    __thrust_exec_check_disable__
    __host__ __device__
    int swap(tuple_leaf& other)
    {
      using thrust::swap;
      swap(this->mutable_get(), other.mutable_get());
      return 0;
    }
};

template<class... Args>
struct tuple_type_list {};

template<size_t i, class... Args>
struct tuple_type_at_impl;

template<size_t i, class Arg0, class... Args>
struct tuple_type_at_impl<i, Arg0, Args...>
{
  using type = typename tuple_type_at_impl<i-1, Args...>::type;
};

template<class Arg0, class... Args>
struct tuple_type_at_impl<0, Arg0,Args...>
{
  using type = Arg0;
};

template<size_t i, class... Args>
using tuple_type_at = typename tuple_type_at_impl<i,Args...>::type;

template<class IndexSequence, class... Args>
class tuple_base;

template<size_t... I, class... Types>
class tuple_base<tuple_index_sequence<I...>, Types...>
  : public tuple_leaf<I,Types>...
{
  public:
    using leaf_types = tuple_type_list<tuple_leaf<I,Types>...>;

    __host__ __device__
    tuple_base() = default;

    __host__ __device__
    tuple_base(const Types&... args)
      : tuple_leaf<I,Types>(args)...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __host__ __device__
    explicit tuple_base(UTypes&&... args)
      : tuple_leaf<I,Types>(std::forward<UTypes>(args))...
    {}


    __host__ __device__
    tuple_base(const tuple_base& other)
      : tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    __host__ __device__
    tuple_base(tuple_base&& other)
      : tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __host__ __device__
    tuple_base(const tuple_base<tuple_index_sequence<I...>,UTypes...>& other)
      : tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,UTypes&&>...
                >::value
             >::type>
    __host__ __device__
    tuple_base(tuple_base<tuple_index_sequence<I...>,UTypes...>&& other)
      : tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    __host__ __device__
    tuple_base& operator=(const tuple_base& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    __host__ __device__
    tuple_base& operator=(tuple_base&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_assignable<Types,const UTypes&>...
                >::value
             >::type>
    __host__ __device__
    tuple_base& operator=(const tuple_base<tuple_index_sequence<I...>,UTypes...>& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_assignable<Types,UTypes&&>...
               >::value
             >::type>
    __host__ __device__
    tuple_base& operator=(tuple_base<tuple_index_sequence<I...>,UTypes...>&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               tuple_and<
                 std::is_assignable<tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __host__ __device__
    tuple_base& operator=(const thrust::pair<UType1,UType2>& p)
    {
      mutable_get<0>() = p.first;
      mutable_get<1>() = p.second;
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               tuple_and<
                 std::is_assignable<tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __host__ __device__
    tuple_base& operator=(thrust::pair<UType1,UType2>&& p)
    {
      mutable_get<0>() = std::move(p.first);
      mutable_get<1>() = std::move(p.second);
      return *this;
    }

    template<size_t i>
    __host__ __device__
    const tuple_leaf<i,tuple_type_at<i,Types...>>& const_leaf() const
    {
      return *this;
    }

    template<size_t i>
    __host__ __device__
    tuple_leaf<i,tuple_type_at<i,Types...>>& mutable_leaf()
    {
      return *this;
    }

    template<size_t i>
    __host__ __device__
    tuple_leaf<i,tuple_type_at<i,Types...>>&& move_leaf() &&
    {
      return std::move(*this);
    }

    __host__ __device__
    void swap(tuple_base& other)
    {
      swallow(tuple_leaf<I,Types>::swap(other)...);
    }

    template<size_t i>
    __host__ __device__
    const tuple_type_at<i,Types...>& const_get() const
    {
      return const_leaf<i>().const_get();
    }

    template<size_t i>
    __host__ __device__
    tuple_type_at<i,Types...>& mutable_get()
    {
      return mutable_leaf<i>().mutable_get();
    }

  private:
    template<class... Args>
    __host__ __device__
    static void swallow(Args&&...) {}
};


} // end detail


template<size_t i, class... UTypes>
__host__ __device__
typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &
  get(thrust::tuple<UTypes...>& t)
{
  return t.template mutable_get<i>();
}


template<size_t i, class... UTypes>
__host__ __device__
const typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &
  get(const thrust::tuple<UTypes...>& t)
{
  return t.template const_get<i>();
}


template<size_t i, class... UTypes>
__host__ __device__
typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &&
  get(thrust::tuple<UTypes...>&& t)
{
  using type = typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type;

  auto&& leaf = static_cast<thrust::detail::tuple_leaf<i,type>&&>(t.base());

  return static_cast<type&&>(leaf.mutable_get());
}


template<class... Types>
class tuple
{
  private:
    using base_type = detail::tuple_base<detail::tuple_make_index_sequence<sizeof...(Types)>, Types...>;
    base_type base_;

    __host__ __device__
    base_type& base()
    {
      return base_;
    }

    __host__ __device__
    const base_type& base() const
    {
      return base_;
    }

  public:
    __host__ __device__
    tuple() : base_{} {};

    __host__ __device__
    explicit tuple(const Types&... args)
      : base_{args...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               detail::tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __host__ __device__
    explicit tuple(UTypes&&... args)
      : base_{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 detail::tuple_and<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             >::type>
    __host__ __device__
    tuple(const tuple<UTypes...>& other)
      : base_{other.base()}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 detail::tuple_and<
                   std::is_constructible<Types,UTypes&&>...
                 >::value
             >::type>
    __host__ __device__
    tuple(tuple<UTypes...>&& other)
      : base_{std::move(other.base())}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               detail::tuple_and<
                 std::is_constructible<detail::tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_constructible<detail::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __host__ __device__
    tuple(const thrust::pair<UType1,UType2>& p)
      : base_{p.first, p.second}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               detail::tuple_and<
                 std::is_constructible<detail::tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_constructible<detail::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __host__ __device__
    tuple(thrust::pair<UType1,UType2>&& p)
      : base_{std::move(p.first), std::move(p.second)}
    {}

    __host__ __device__
    tuple(const tuple& other)
      : base_{other.base()}
    {}

    __host__ __device__
    tuple(tuple&& other)
      : base_{std::move(other.base())}
    {}

    __host__ __device__
    tuple& operator=(const tuple& other)
    {
      base().operator=(other.base());
      return *this;
    }

    __host__ __device__
    tuple& operator=(tuple&& other)
    {
      base().operator=(std::move(other.base()));
      return *this;
    }

    // XXX needs enable_if
    template<class... UTypes>
    __host__ __device__
    tuple& operator=(const tuple<UTypes...>& other)
    {
      base().operator=(other.base());
      return *this;
    }

    // XXX needs enable_if
    template<class... UTypes>
    __host__ __device__
    tuple& operator=(tuple<UTypes...>&& other)
    {
      base().operator=(other.base());
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               detail::tuple_and<
                 std::is_assignable<detail::tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<detail::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __host__ __device__
    tuple& operator=(const thrust::pair<UType1,UType2>& p)
    {
      base().operator=(p);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               detail::tuple_and<
                 std::is_assignable<detail::tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<detail::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __host__ __device__
    tuple& operator=(thrust::pair<UType1,UType2>&& p)
    {
      base().operator=(std::move(p));
      return *this;
    }

    __host__ __device__
    void swap(tuple& other)
    {
      base().swap(other.base());
    }

  private:
    template<class... UTypes>
    friend class tuple;

    template<size_t i>
    __host__ __device__
    const typename thrust::tuple_element<i,tuple>::type& const_get() const
    {
      return base().template const_get<i>();
    }

    template<size_t i>
    __host__ __device__
    typename thrust::tuple_element<i,tuple>::type& mutable_get()
    {
      return base().template mutable_get<i>();
    }

  public:
    template<size_t i, class... UTypes>
    friend __host__ __device__
    typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &
    thrust::get(thrust::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __host__ __device__
    const typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &
    thrust::get(const thrust::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __host__ __device__
    typename thrust::tuple_element<i, thrust::tuple<UTypes...>>::type &&
    thrust::get(thrust::tuple<UTypes...>&& t);
};


template<>
class tuple<>
{
  public:
    __host__ __device__
    void swap(tuple&){}
};


template<class... Types>
__host__ __device__
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
__host__ __device__
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
__host__ __device__
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
__host__ __device__
thrust::tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return thrust::tuple<Args&&...>(std::forward<Args>(args)...);
}


namespace detail
{


struct tuple_ignore_t
{
  template<class T>
  __host__ __device__
  const tuple_ignore_t operator=(T&&) const
  {
    return *this;
  }
};


} // end detail


constexpr detail::tuple_ignore_t ignore{};


namespace detail
{


template<size_t I, class T, class... Types>
struct tuple_find_exactly_one_impl;


template<size_t I, class T, class U, class... Types>
struct tuple_find_exactly_one_impl<I,T,U,Types...> : tuple_find_exactly_one_impl<I+1, T, Types...> {};


template<size_t I, class T, class... Types>
struct tuple_find_exactly_one_impl<I,T,T,Types...> : std::integral_constant<size_t, I>
{
  static_assert(tuple_find_exactly_one_impl<I,T,Types...>::value == -1, "type can only occur once in type list");
};


template<size_t I, class T>
struct tuple_find_exactly_one_impl<I,T> : std::integral_constant<int, -1> {};


template<class T, class... Types>
struct tuple_find_exactly_one : tuple_find_exactly_one_impl<0,T,Types...>
{
  static_assert(tuple_find_exactly_one::value != -1, "type not found in type list");
};


} // end detail


template<class T, class... Types>
__host__ __device__
T& get(thrust::tuple<Types...>& t)
{
  return thrust::get<thrust::detail::tuple_find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__host__ __device__
const T& get(const thrust::tuple<Types...>& t)
{
  return thrust::get<thrust::detail::tuple_find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__host__ __device__
T&& get(thrust::tuple<Types...>&& t)
{
  return thrust::get<thrust::detail::tuple_find_exactly_one<T,Types...>::value>(std::move(t));
}


// implement relational operators
namespace detail
{


__host__ __device__
  inline bool tuple_all()
{
  return true;
}


__host__ __device__
  inline bool tuple_all(bool t)
{
  return t;
}


template<typename... Bools>
__host__ __device__
  bool tuple_all(bool t, Bools... ts)
{
  return t && detail::tuple_all(ts...);
}


} // end detail


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


namespace detail
{


template<class... TTypes, class... UTypes, size_t... I>
__host__ __device__
  bool tuple_eq(const tuple<TTypes...>& t, const tuple<UTypes...>& u, detail::tuple_index_sequence<I...>)
{
  return detail::tuple_all((thrust::get<I>(t) == thrust::get<I>(u))...);
}


} // end detail


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return detail::tuple_eq(t, u, detail::tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


namespace detail
{


template<class... TTypes, class... UTypes>
__host__ __device__
  bool tuple_lt(const tuple<TTypes...>& t, const tuple<UTypes...>& u, tuple_index_sequence<>)
{
  return false;
}


template<size_t I, class... TTypes, class... UTypes, size_t... Is>
__host__ __device__
  bool tuple_lt(const tuple<TTypes...>& t, const tuple<UTypes...>& u, tuple_index_sequence<I, Is...>)
{
  return (   thrust::get<I>(t) < thrust::get<I>(u)
          || (!(thrust::get<I>(u) < thrust::get<I>(t))
              && detail::tuple_lt(t, u, typename tuple_make_index_sequence_impl<I+1, tuple_index_sequence<>, sizeof...(TTypes)>::type{})));
}


} // end detail


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return detail::tuple_lt(t, u, detail::tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator!=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t == u);
}


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return u < t;
}


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(u < t);
}


template<class... TTypes, class... UTypes>
__host__ __device__
  bool operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t < u);
}


} // end thrust


