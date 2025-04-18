# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from yesense_imu/YesenseImuGnssData.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import yesense_imu.msg

class YesenseImuGnssData(genpy.Message):
  _md5sum = "af4901e8965d58039a1c91b44d48619c"
  _type = "yesense_imu/YesenseImuGnssData"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """yesense_imu/YesenseImuMasterGnssData master
yesense_imu/YesenseImuSlaveGnssData slave
================================================================================
MSG: yesense_imu/YesenseImuMasterGnssData
yesense_imu/YesenseImuUtcTime   utc_time
yesense_imu/YesenseImuLocation  location
yesense_imu/YesenseImuLocation  location_error
yesense_imu/YesenseImuVelocity  vel
float32 speed
float32 yaw
uint8 status
uint8 star_cnt
float32 p_dop
uint8 site_id
================================================================================
MSG: yesense_imu/YesenseImuUtcTime
uint16 year
uint8 month
uint8 date
uint8 hour
uint8 min
uint8 sec
uint32 ms
================================================================================
MSG: yesense_imu/YesenseImuLocation
float64 longtidue
float64 latitude
float32 altidue
================================================================================
MSG: yesense_imu/YesenseImuVelocity
float32 v_e
float32 v_n
float32 v_u

================================================================================
MSG: yesense_imu/YesenseImuSlaveGnssData
float32 dual_ant_yaw
float32 dual_ant_yaw_error
float32 dual_ant_baseline_len"""
  __slots__ = ['master','slave']
  _slot_types = ['yesense_imu/YesenseImuMasterGnssData','yesense_imu/YesenseImuSlaveGnssData']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       master,slave

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(YesenseImuGnssData, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.master is None:
        self.master = yesense_imu.msg.YesenseImuMasterGnssData()
      if self.slave is None:
        self.slave = yesense_imu.msg.YesenseImuSlaveGnssData()
    else:
      self.master = yesense_imu.msg.YesenseImuMasterGnssData()
      self.slave = yesense_imu.msg.YesenseImuSlaveGnssData()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_H5BI2df2d6f2BfB3f().pack(_x.master.utc_time.year, _x.master.utc_time.month, _x.master.utc_time.date, _x.master.utc_time.hour, _x.master.utc_time.min, _x.master.utc_time.sec, _x.master.utc_time.ms, _x.master.location.longtidue, _x.master.location.latitude, _x.master.location.altidue, _x.master.location_error.longtidue, _x.master.location_error.latitude, _x.master.location_error.altidue, _x.master.vel.v_e, _x.master.vel.v_n, _x.master.vel.v_u, _x.master.speed, _x.master.yaw, _x.master.status, _x.master.star_cnt, _x.master.p_dop, _x.master.site_id, _x.slave.dual_ant_yaw, _x.slave.dual_ant_yaw_error, _x.slave.dual_ant_baseline_len))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.master is None:
        self.master = yesense_imu.msg.YesenseImuMasterGnssData()
      if self.slave is None:
        self.slave = yesense_imu.msg.YesenseImuSlaveGnssData()
      end = 0
      _x = self
      start = end
      end += 90
      (_x.master.utc_time.year, _x.master.utc_time.month, _x.master.utc_time.date, _x.master.utc_time.hour, _x.master.utc_time.min, _x.master.utc_time.sec, _x.master.utc_time.ms, _x.master.location.longtidue, _x.master.location.latitude, _x.master.location.altidue, _x.master.location_error.longtidue, _x.master.location_error.latitude, _x.master.location_error.altidue, _x.master.vel.v_e, _x.master.vel.v_n, _x.master.vel.v_u, _x.master.speed, _x.master.yaw, _x.master.status, _x.master.star_cnt, _x.master.p_dop, _x.master.site_id, _x.slave.dual_ant_yaw, _x.slave.dual_ant_yaw_error, _x.slave.dual_ant_baseline_len,) = _get_struct_H5BI2df2d6f2BfB3f().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_H5BI2df2d6f2BfB3f().pack(_x.master.utc_time.year, _x.master.utc_time.month, _x.master.utc_time.date, _x.master.utc_time.hour, _x.master.utc_time.min, _x.master.utc_time.sec, _x.master.utc_time.ms, _x.master.location.longtidue, _x.master.location.latitude, _x.master.location.altidue, _x.master.location_error.longtidue, _x.master.location_error.latitude, _x.master.location_error.altidue, _x.master.vel.v_e, _x.master.vel.v_n, _x.master.vel.v_u, _x.master.speed, _x.master.yaw, _x.master.status, _x.master.star_cnt, _x.master.p_dop, _x.master.site_id, _x.slave.dual_ant_yaw, _x.slave.dual_ant_yaw_error, _x.slave.dual_ant_baseline_len))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.master is None:
        self.master = yesense_imu.msg.YesenseImuMasterGnssData()
      if self.slave is None:
        self.slave = yesense_imu.msg.YesenseImuSlaveGnssData()
      end = 0
      _x = self
      start = end
      end += 90
      (_x.master.utc_time.year, _x.master.utc_time.month, _x.master.utc_time.date, _x.master.utc_time.hour, _x.master.utc_time.min, _x.master.utc_time.sec, _x.master.utc_time.ms, _x.master.location.longtidue, _x.master.location.latitude, _x.master.location.altidue, _x.master.location_error.longtidue, _x.master.location_error.latitude, _x.master.location_error.altidue, _x.master.vel.v_e, _x.master.vel.v_n, _x.master.vel.v_u, _x.master.speed, _x.master.yaw, _x.master.status, _x.master.star_cnt, _x.master.p_dop, _x.master.site_id, _x.slave.dual_ant_yaw, _x.slave.dual_ant_yaw_error, _x.slave.dual_ant_baseline_len,) = _get_struct_H5BI2df2d6f2BfB3f().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_H5BI2df2d6f2BfB3f = None
def _get_struct_H5BI2df2d6f2BfB3f():
    global _struct_H5BI2df2d6f2BfB3f
    if _struct_H5BI2df2d6f2BfB3f is None:
        _struct_H5BI2df2d6f2BfB3f = struct.Struct("<H5BI2df2d6f2BfB3f")
    return _struct_H5BI2df2d6f2BfB3f
