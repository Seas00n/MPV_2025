// Generated by gencpp from file yesense_imu/YesenseImuSensorData.msg
// DO NOT EDIT!


#ifndef YESENSE_IMU_MESSAGE_YESENSEIMUSENSORDATA_H
#define YESENSE_IMU_MESSAGE_YESENSEIMUSENSORDATA_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Accel.h>
#include <yesense_imu/YesenseImuQuaternion.h>
#include <yesense_imu/YesenseImuEulerAngle.h>
#include <yesense_imu/YesenseImuLocation.h>

namespace yesense_imu
{
template <class ContainerAllocator>
struct YesenseImuSensorData_
{
  typedef YesenseImuSensorData_<ContainerAllocator> Type;

  YesenseImuSensorData_()
    : temperature(0.0)
    , sample_timestamp(0)
    , sync_timestamp(0)
    , accel()
    , quaternion()
    , eulerAngle()
    , location()  {
    }
  YesenseImuSensorData_(const ContainerAllocator& _alloc)
    : temperature(0.0)
    , sample_timestamp(0)
    , sync_timestamp(0)
    , accel(_alloc)
    , quaternion(_alloc)
    , eulerAngle(_alloc)
    , location(_alloc)  {
  (void)_alloc;
    }



   typedef float _temperature_type;
  _temperature_type temperature;

   typedef uint32_t _sample_timestamp_type;
  _sample_timestamp_type sample_timestamp;

   typedef uint32_t _sync_timestamp_type;
  _sync_timestamp_type sync_timestamp;

   typedef  ::geometry_msgs::Accel_<ContainerAllocator>  _accel_type;
  _accel_type accel;

   typedef  ::yesense_imu::YesenseImuQuaternion_<ContainerAllocator>  _quaternion_type;
  _quaternion_type quaternion;

   typedef  ::yesense_imu::YesenseImuEulerAngle_<ContainerAllocator>  _eulerAngle_type;
  _eulerAngle_type eulerAngle;

   typedef  ::yesense_imu::YesenseImuLocation_<ContainerAllocator>  _location_type;
  _location_type location;





  typedef boost::shared_ptr< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> const> ConstPtr;

}; // struct YesenseImuSensorData_

typedef ::yesense_imu::YesenseImuSensorData_<std::allocator<void> > YesenseImuSensorData;

typedef boost::shared_ptr< ::yesense_imu::YesenseImuSensorData > YesenseImuSensorDataPtr;
typedef boost::shared_ptr< ::yesense_imu::YesenseImuSensorData const> YesenseImuSensorDataConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator1> & lhs, const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator2> & rhs)
{
  return lhs.temperature == rhs.temperature &&
    lhs.sample_timestamp == rhs.sample_timestamp &&
    lhs.sync_timestamp == rhs.sync_timestamp &&
    lhs.accel == rhs.accel &&
    lhs.quaternion == rhs.quaternion &&
    lhs.eulerAngle == rhs.eulerAngle &&
    lhs.location == rhs.location;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator1> & lhs, const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace yesense_imu

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6d4626fa769075113f501bc181b31122";
  }

  static const char* value(const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6d4626fa76907511ULL;
  static const uint64_t static_value2 = 0x3f501bc181b31122ULL;
};

template<class ContainerAllocator>
struct DataType< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "yesense_imu/YesenseImuSensorData";
  }

  static const char* value(const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32 temperature\n"
"uint32 sample_timestamp\n"
"uint32 sync_timestamp\n"
"geometry_msgs/Accel accel\n"
"yesense_imu/YesenseImuQuaternion quaternion\n"
"yesense_imu/YesenseImuEulerAngle eulerAngle\n"
"yesense_imu/YesenseImuLocation location\n"
"================================================================================\n"
"MSG: geometry_msgs/Accel\n"
"# This expresses acceleration in free space broken into its linear and angular parts.\n"
"Vector3  linear\n"
"Vector3  angular\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"================================================================================\n"
"MSG: yesense_imu/YesenseImuQuaternion\n"
"float32 q0\n"
"float32 q1\n"
"float32 q2\n"
"float32 q3\n"
"\n"
"================================================================================\n"
"MSG: yesense_imu/YesenseImuEulerAngle\n"
"float32 roll\n"
"float32 pitch\n"
"float32 yaw\n"
"\n"
"================================================================================\n"
"MSG: yesense_imu/YesenseImuLocation\n"
"float64 longtidue\n"
"float64 latitude\n"
"float32 altidue\n"
;
  }

  static const char* value(const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.temperature);
      stream.next(m.sample_timestamp);
      stream.next(m.sync_timestamp);
      stream.next(m.accel);
      stream.next(m.quaternion);
      stream.next(m.eulerAngle);
      stream.next(m.location);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct YesenseImuSensorData_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::yesense_imu::YesenseImuSensorData_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::yesense_imu::YesenseImuSensorData_<ContainerAllocator>& v)
  {
    s << indent << "temperature: ";
    Printer<float>::stream(s, indent + "  ", v.temperature);
    s << indent << "sample_timestamp: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.sample_timestamp);
    s << indent << "sync_timestamp: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.sync_timestamp);
    s << indent << "accel: ";
    s << std::endl;
    Printer< ::geometry_msgs::Accel_<ContainerAllocator> >::stream(s, indent + "  ", v.accel);
    s << indent << "quaternion: ";
    s << std::endl;
    Printer< ::yesense_imu::YesenseImuQuaternion_<ContainerAllocator> >::stream(s, indent + "  ", v.quaternion);
    s << indent << "eulerAngle: ";
    s << std::endl;
    Printer< ::yesense_imu::YesenseImuEulerAngle_<ContainerAllocator> >::stream(s, indent + "  ", v.eulerAngle);
    s << indent << "location: ";
    s << std::endl;
    Printer< ::yesense_imu::YesenseImuLocation_<ContainerAllocator> >::stream(s, indent + "  ", v.location);
  }
};

} // namespace message_operations
} // namespace ros

#endif // YESENSE_IMU_MESSAGE_YESENSEIMUSENSORDATA_H
