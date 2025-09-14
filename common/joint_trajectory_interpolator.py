from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si

def joint_distance(start_joints: np.ndarray, end_joints: np.ndarray) -> float:
    """计算两个关节位置之间的欧几里得距离"""
    start_joints = np.array(start_joints)
    end_joints = np.array(end_joints)
    return np.linalg.norm(end_joints - start_joints)

class JointTrajectoryInterpolator:
    """关节轨迹线性插值器"""
    
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        """
        初始化关节轨迹插值器
        
        Args:
            times: 时间数组
            joints: 关节位置数组 (N, 7)
        """
        assert len(times) >= 1
        assert len(joints) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)

        if len(times) == 1:
            # 单步插值的特殊处理
            self.single_step = True
            self._times = times
            self._joints = joints
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            
            # 使用三次样条插值
            self.joint_interp = si.interp1d(times, joints, 
                axis=0, assume_sorted=True, kind='cubic')
    
    @property
    def times(self) -> np.ndarray:
        """获取时间数组"""
        if self.single_step:
            return self._times
        else:
            return self.joint_interp.x
    
    @property
    def joints(self) -> np.ndarray:
        """获取关节位置数组"""
        if self.single_step:
            return self._joints
        else:
            return self.joint_interp.y

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        """
        修剪插值器到指定时间范围
        
        Args:
            start_t: 开始时间
            end_t: 结束时间
            
        Returns:
            新的关节轨迹插值器
        """
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # 移除重复值
        all_times = np.unique(all_times)
        # 插值
        all_joints = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)
    
    def drive_to_waypoint(self, 
            joints: np.ndarray, 
            time: float, 
            curr_time: float,
            max_joint_speed: float = np.inf
        ) -> "JointTrajectoryInterpolator":
        """
        驱动到目标关节位置
        
        Args:
            joints: 目标关节位置 (7,)
            time: 目标时间
            curr_time: 当前时间
            max_joint_speed: 最大关节速度 (rad/s)
            
        Returns:
            新的关节轨迹插值器
        """
        assert max_joint_speed > 0
        time = max(time, curr_time)
        
        curr_joints = self(curr_time)
        joint_dist = joint_distance(curr_joints, joints)
        min_duration = joint_dist / max_joint_speed
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # 插入新的关节位置
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints_array = np.append(trimmed_interp.joints, [joints], axis=0)

        # 创建新的插值器
        final_interp = JointTrajectoryInterpolator(times, joints_array)
        return final_interp

    def schedule_waypoint(self,
            joints: np.ndarray, 
            time: float, 
            max_joint_speed: float = np.inf,
            curr_time: float = None,
            last_waypoint_time: float = None
        ) -> "JointTrajectoryInterpolator":
        """
        调度目标关节位置
        
        Args:
            joints: 目标关节位置 (7,)
            time: 目标时间
            max_joint_speed: 最大关节速度 (rad/s)
            curr_time: 当前时间
            last_waypoint_time: 最后一个路径点时间
            
        Returns:
            新的关节轨迹插值器
        """
        assert max_joint_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        # 修剪当前插值器到 curr_time 和 last_waypoint_time 之间
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # 如果插入时间早于当前时间，不对插值器产生任何影响
                return self
            # 现在 curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # 如果 last_waypoint_time 早于 start_time，使用 start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        
        # 约束条件:
        # start_time <= end_time <= time
        # curr_time <= start_time
        # curr_time <= time
        
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # 在此之后，trimmed_interp 中的所有路径点都在 start_time 和 end_time 之间
        # 并且早于 time

        # 确定速度
        duration = time - end_time
        end_joints = trimmed_interp(end_time)
        joint_dist = joint_distance(joints, end_joints)
        min_duration = joint_dist / max_joint_speed
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # 插入新的关节位置
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints_array = np.append(trimmed_interp.joints, [joints], axis=0)

        # 创建新的插值器
        final_interp = JointTrajectoryInterpolator(times, joints_array)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        在指定时间插值关节位置
        
        Args:
            t: 时间点或时间数组
            
        Returns:
            插值后的关节位置
        """
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        joints = np.zeros((len(t), 7))
        if self.single_step:
            joints[:] = self._joints[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            joints = self.joint_interp(t)

        if is_single:
            joints = joints[0]
        return joints
