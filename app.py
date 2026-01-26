"""
PX4 Log Analyzer - Расширенная версия
Больше параметров, исправленная высота, поддержка GPS
"""
import os
import uuid
import json
import math
import re
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file
import pandas as pd
import plotly
import plotly.graph_objects as go
import pyulog

# ========== КОНФИГУРАЦИЯ ==========
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# ========== РАСШИРЕННЫЙ АНАЛИЗАТОР ==========
class EnhancedAnalyzer:
    """Анализатор с поддержкой множества параметров"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.ulog = None
        self.topics_data = {}
        self.has_gps = False
        self.gps_coords = []
        
    def analyze(self):
        """Полный анализ лога"""
        try:
            self.ulog = pyulog.ULog(self.file_path)
            
            # Собираем все топики
            topics = []
            for data in self.ulog.data_list:
                topics.append(data.name)
            
            # Извлекаем данные из всех ключевых топиков
            key_topics = [
                'vehicle_gps_position', 'vehicle_global_position', 'battery_status', 'vehicle_attitude',
                'vehicle_local_position', 'sensor_combined', 'actuator_outputs',
                'estimator_status', 'vehicle_angular_velocity', 'vehicle_acceleration',
                'rc_channels', 'cpuload', 'vehicle_magnetometer', 'vehicle_air_data'
            ]
            
            for topic in key_topics:
                df = self._extract_data(topic)
                if df is not None and not df.empty:
                    self.topics_data[topic] = df
            
            # Проверяем наличие GPS данных в разных топиках
            gps_topics = ['vehicle_gps_position', 'vehicle_global_position', 'vehicle_local_position']
            for gps_topic in gps_topics:
                if gps_topic in self.topics_data:
                    self._extract_gps_coords(self.topics_data[gps_topic], gps_topic)
                    if self.has_gps:  # Если нашли GPS, останавливаем поиск
                        break
            
            # Если ключевых топиков мало, ищем любые данные
            if len(self.topics_data) < 5:
                for topic in topics:
                    if topic not in self.topics_data:
                        df = self._extract_data(topic)
                        if df is not None and len(df) > 10:
                            self.topics_data[topic] = df
                            if len(self.topics_data) >= 8:
                                break
            
            duration = (self.ulog.last_timestamp - self.ulog.start_timestamp) / 1e6
            
            return {
                'success': True,
                'duration': duration,
                'start_time': datetime.fromtimestamp(self.ulog.start_timestamp / 1e6),
                'topics_found': list(self.topics_data.keys()),
                'topics_count': len(topics),
                'has_gps': self.has_gps,
                'gps_points': len(self.gps_coords)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_data(self, topic_name):
        """Извлечение данных из топика"""
        for data in self.ulog.data_list:
            if data.name == topic_name:
                df = pd.DataFrame(data.data)
                
                if df.empty:
                    return None
                
                # Добавляем время
                if 'timestamp' in df.columns:
                    df['time_sec'] = (df['timestamp'] - self.ulog.start_timestamp) / 1e6
                
                # Оптимизация
                if len(df) > 1000:
                    step = max(1, len(df) // 500)
                    df = df.iloc[::step].copy()
                
                return df
        
        return None
    
    def _extract_gps_coords(self, df, topic_name):
        """Извлечение GPS координат из любого подходящего топика"""
        lat_col = None
        lon_col = None
        alt_col = None
        
        # Определяем возможные названия колонок для разных топиков
        if topic_name == 'vehicle_gps_position':
            # Стандартный GPS топик PX4
            if 'lat' in df.columns:
                lat_col = 'lat'
            elif 'latitude' in df.columns:
                lat_col = 'latitude'
                
            if 'lon' in df.columns:
                lon_col = 'lon'
            elif 'longitude' in df.columns:
                lon_col = 'longitude'
                
            if 'alt' in df.columns:
                alt_col = 'alt'
            elif 'altitude' in df.columns:
                alt_col = 'altitude'
                
        elif topic_name == 'vehicle_global_position':
            # Глобальная позиция (обычно используется в навигации)
            if 'lat' in df.columns:
                lat_col = 'lat'
            elif 'latitude' in df.columns:
                lat_col = 'latitude'
                
            if 'lon' in df.columns:
                lon_col = 'lon'
            elif 'longitude' in df.columns:
                lon_col = 'longitude'
                
            if 'alt' in df.columns:
                alt_col = 'alt'
            elif 'altitude' in df.columns:
                alt_col = 'altitude'
            elif 'alt_ellipsoid' in df.columns:
                alt_col = 'alt_ellipsoid'
                
        elif topic_name == 'vehicle_local_position':
            # Локальная позиция (может иметь глобальные координаты)
            if 'ref_lat' in df.columns and 'ref_lon' in df.columns:
                # Если есть reference point, можем вычислить абсолютные координаты
                if len(df) > 0:
                    ref_lat = df.iloc[0]['ref_lat'] / 1e7
                    ref_lon = df.iloc[0]['ref_lon'] / 1e7
                    # Для простоты пока не обрабатываем локальные координаты
                    pass

        if lat_col and lon_col:
            self.has_gps = True
            
            # Берем каждую 5-ю точку для оптимизации
            step = max(1, len(df) // 200)
            for i in range(0, min(len(df), 2000), step):
                try:
                    lat_raw = float(df.iloc[i][lat_col])
                    lon_raw = float(df.iloc[i][lon_col])
                    
                    # Проверяем валидность координат
                    if abs(lat_raw) < 1e-6 and abs(lon_raw) < 1e-6:
                        continue
                    
                    # PX4 хранит координаты в разных форматах:
                    # 1. vehicle_gps_position: int32_t в 1e7 формате (deg * 1e7)
                    # 2. vehicle_global_position: double в градусах
                    
                    if topic_name == 'vehicle_gps_position':
                        # GPS топик: координаты в 1e7 формате
                        lat = lat_raw / 1e7
                        lon = lon_raw / 1e7
                    else:
                        # Другие топики: обычно в градусах
                        if abs(lat_raw) > 180 or abs(lon_raw) > 180:
                            # Если значения слишком большие для градусов, это 1e7 формат
                            lat = lat_raw / 1e7
                            lon = lon_raw / 1e7
                        else:
                            lat = lat_raw
                            lon = lon_raw
                    
                    # Получаем высоту
                    alt = 0
                    if alt_col and alt_col in df.columns:
                        alt_raw = float(df.iloc[i][alt_col])
                        if topic_name == 'vehicle_gps_position':
                            # GPS топик: высота в миллиметрах
                            alt = alt_raw / 1000  # мм → метры
                        else:
                            # Другие топики: обычно в метрах
                            if abs(alt_raw) > 100000:  # Если слишком большое, возможно мм
                                alt = alt_raw / 1000
                            else:
                                alt = alt_raw
                    
                    # Проверяем, что координаты валидны
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        self.gps_coords.append([lat, lon, alt])
                        
                except (ValueError, TypeError, KeyError, IndexError) as e:
                    continue
            
            # Если нашли точки, сортируем по времени (если есть timestamp)
            if self.gps_coords and 'timestamp' in df.columns:
                try:
                    # Создаем DataFrame для сортировки
                    gps_df = pd.DataFrame(self.gps_coords, columns=['lat', 'lon', 'alt'])
                    
                    # Берем соответствующие временные метки
                    indices = list(range(0, min(len(df), 2000), step))
                    if len(indices) >= len(gps_df):
                        gps_df['timestamp'] = [df.iloc[i]['timestamp'] for i in indices[:len(gps_df)]]
                        gps_df = gps_df.sort_values('timestamp')
                        self.gps_coords = gps_df[['lat', 'lon', 'alt']].values.tolist()
                except Exception as e:
                    pass
    
    def get_best_parameters(self):
        """Находит лучшие параметры для отображения"""
        parameters = []
        
        # 1. Позиция и навигация
        if 'vehicle_gps_position' in self.topics_data:
            df = self.topics_data['vehicle_gps_position']
            
            # Высота GPS (обычно положительная)
            if 'alt' in df.columns:
                values = df['alt'].dropna().values
                if len(values) > 0:
                    # GPS высота в миллиметрах → метры
                    parameters.append(self._create_param(
                        'gps_alt', 'Высота GPS', 'м', '📈', '#3498db', values / 1000
                    ))
            
            # Скорость (м/с → км/ч)
            if 'vel_m_s' in df.columns:
                values = df['vel_m_s'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'gps_speed', 'Скорость GPS', 'км/ч', '⚡', '#2ecc71', values * 3.6
                    ))
            
            # Количество спутников
            if 'satellites_used' in df.columns:
                values = df['satellites_used'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'gps_satellites', 'Спутники', 'шт', '🛰️', '#9b59b6', values
                    ))
            
            # Fix type (качество GPS)
            if 'fix_type' in df.columns:
                values = df['fix_type'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'gps_fix_type', 'Качество GPS', '', '📍', '#e74c3c', values
                    ))
        
        # 2. Глобальная позиция
        if 'vehicle_global_position' in self.topics_data:
            df = self.topics_data['vehicle_global_position']
            
            # Высота (метры)
            if 'alt' in df.columns:
                values = df['alt'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'global_alt', 'Высота (глоб.)', 'м', '🗺️', '#3498db', values
                    ))
            
            # Скорость
            if 'vel_n' in df.columns and 'vel_e' in df.columns:
                vel_n = df['vel_n'].dropna().values
                vel_e = df['vel_e'].dropna().values
                if len(vel_n) > 0 and len(vel_e) > 0:
                    # Вычисляем горизонтальную скорость
                    speed = np.sqrt(vel_n**2 + vel_e**2)
                    parameters.append(self._create_param(
                        'global_speed', 'Скорость (глоб.)', 'м/с', '🌐', '#2ecc71', speed
                    ))
        
        # 3. Локальная позиция (инвертируем высоту для понятности)
        if 'vehicle_local_position' in self.topics_data:
            df = self.topics_data['vehicle_local_position']
            
            if 'z' in df.columns:
                values = df['z'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'local_z', 'Высота (локальная)', 'м', '📏', '#f39c12', -values  # Инвертируем!
                    ))
            
            # Скорости по осям
            for axis, name, color in [('vx', 'Скорость X', '#1abc9c'), 
                                     ('vy', 'Скорость Y', '#16a085'),
                                     ('vz', 'Скорость Z', '#27ae60')]:
                if axis in df.columns:
                    values = df[axis].dropna().values
                    if len(values) > 0:
                        parameters.append(self._create_param(
                            f'local_{axis}', name, 'м/с', '↗️', color, values
                        ))
            
            # Позиция по X, Y
            if 'x' in df.columns and 'y' in df.columns:
                x_vals = df['x'].dropna().values
                y_vals = df['y'].dropna().values
                if len(x_vals) > 0 and len(y_vals) > 0:
                    # Вычисляем горизонтальное смещение
                    dist = np.sqrt(x_vals**2 + y_vals**2)
                    parameters.append(self._create_param(
                        'local_distance', 'Дистанция', 'м', '📐', '#8e44ad', dist
                    ))
        
        # 4. Ориентация (радианы → градусы)
        if 'vehicle_attitude' in self.topics_data:
            df = self.topics_data['vehicle_attitude']
            
            for field, name, color in [('roll', 'Крен', '#e74c3c'),
                                      ('pitch', 'Тангаж', '#8e44ad'),
                                      ('yaw', 'Рыскание', '#d35400')]:
                if field in df.columns:
                    values = df[field].dropna().values
                    if len(values) > 0:
                        parameters.append(self._create_param(
                            f'attitude_{field}', name, '°', '✈️', color,
                            [math.degrees(v) for v in values]
                        ))
        
        # 5. Батарея
        if 'battery_status' in self.topics_data:
            df = self.topics_data['battery_status']
            
            if 'voltage_v' in df.columns:
                values = df['voltage_v'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'battery_voltage', 'Напряжение', 'В', '🔋', '#c0392b', values
                    ))
            
            if 'current_a' in df.columns:
                values = df['current_a'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'battery_current', 'Ток', 'А', '⚡', '#d35400', values
                    ))
              
            if 'remaining' in df.columns:
                values = df['remaining'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'battery_remaining', 'Заряд', '%', '🔌', '#2ecc71', values * 100
                    ))
        
        # 6. Датчики IMU
        if 'sensor_combined' in self.topics_data:
            df = self.topics_data['sensor_combined']
            
            # Ускорения
            for i, axis in enumerate(['x', 'y', 'z']):
                col = f'accelerometer_m_s2[{i}]'
                if col in df.columns:
                    values = df[col].dropna().values
                    if len(values) > 0:
                        parameters.append(self._create_param(
                            f'accel_{axis}', f'Ускорение {axis.upper()}', 'м/с²', '📡', 
                            '#8e44ad', values
                        ))
            
            # Гироскоп
            for i, axis in enumerate(['x', 'y', 'z']):
                col = f'gyro_rad[{i}]'
                if col in df.columns:
                    values = df[col].dropna().values
                    if len(values) > 0:
                        # Рад/с → град/с
                        parameters.append(self._create_param(
                            f'gyro_{axis}', f'Гироскоп {axis.upper()}', '°/с', '🔄', 
                            '#16a085', [math.degrees(v) for v in values]
                        ))
        
        # 7. RC каналы
        if 'rc_channels' in self.topics_data:
            df = self.topics_data['rc_channels']
            
            for i in range(4):  # Первые 4 канала
                col = f'channels[{i}]'
                if col in df.columns:
                    values = df[col].dropna().values
                    if len(values) > 0:
                        parameters.append(self._create_param(
                            f'rc_{i+1}', f'RC Канал {i+1}', '', '🎮',
                            '#9b59b6', values
                        ))
        
        # 8. Моторы
        if 'actuator_outputs' in self.topics_data:
            df = self.topics_data['actuator_outputs']
            
            for i in range(4):  # Первые 4 мотора
                col = f'output[{i}]'
                if col in df.columns:
                    values = df[col].dropna().values
                    if len(values) > 0:
                        parameters.append(self._create_param(
                            f'motor_{i+1}', f'Мотор {i+1}', '', '⚙️',
                            '#7f8c8d', values
                        ))
        
        # 9. Температура батареи
        if 'battery_status' in self.topics_data:
            df = self.topics_data['battery_status']
            
            if 'temperature' in df.columns:
                values = df['temperature'].dropna().values
                if len(values) > 0:
                    parameters.append(self._create_param(
                        'battery_temp', 'Темп. батареи', '°C', '🌡️',
                        '#e74c3c', values
                    ))
        
        # Если все еще мало параметров, ищем любые числовые данные
        if len(parameters) < 8:
            for topic, df in self.topics_data.items():
                if topic in ['vehicle_gps_position', 'battery_status', 'vehicle_attitude',
                           'vehicle_local_position', 'sensor_combined', 'actuator_outputs',
                           'rc_channels']:
                    continue  # Уже обработали
                
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'time_sec']]
                
                for col in numeric_cols[:2]:  # Берем первые 2 колонки
                    values = df[col].dropna().values
                    if len(values) > 10:
                        parameters.append(self._create_param(
                            f'{topic}_{col}', f'{topic}: {col}', 'ед.', '📊',
                            '#7f8c8d', values
                        ))
                    
                    if len(parameters) >= 15:  # Максимум 15 параметров
                        break
                if len(parameters) >= 15:
                    break
        
        # Добавляем статистику и удаляем values
        for param in parameters:
            param['stats'] = {
                'current': float(param['values'][-1]) if len(param['values']) > 0 else 0,
                'min': float(min(param['values'])) if len(param['values']) > 0 else 0,
                'max': float(max(param['values'])) if len(param['values']) > 0 else 0,
                'avg': float(sum(param['values']) / len(param['values'])) if len(param['values']) > 0 else 0,
                'count': len(param['values'])
            }
            del param['values']
        
        # Группируем по категориям
        categories = {
            '📍 Позиция и навигация': [],
            '✈️ Ориентация': [],
            '🔋 Энергия': [],
            '📡 Датчики': [],
            '⚙️ Управление': [],
            '📊 Другие параметры': []
        }
        
        for param in parameters:
            name = param['name']
            if any(word in name for word in ['Высота', 'Скорость', 'Спутники', 'Точность', 'Дистанция', 'GPS', 'глоб', 'локаль']):
                categories['📍 Позиция и навигация'].append(param)
            elif any(word in name for word in ['Крен', 'Тангаж', 'Рыскание']):
                categories['✈️ Ориентация'].append(param)
            elif any(word in name for word in ['Напряжение', 'Ток', 'Заряд', 'Температура', 'батареи']):
                categories['🔋 Энергия'].append(param)
            elif any(word in name for word in ['Ускорение', 'Гироскоп', 'Магнитометр']):
                categories['📡 Датчики'].append(param)
            elif any(word in name for word in ['Мотор', 'Канал', 'RC']):
                categories['⚙️ Управление'].append(param)
            else:
                categories['📊 Другие параметры'].append(param)
        
        # Убираем пустые категории
        categories = {k: v for k, v in categories.items() if v}
        
        return {
            'all': parameters,
            'categories': categories,
            'total_count': len(parameters)
        }
    
    def _create_param(self, param_id, name, unit, icon, color, values):
        """Создание объекта параметра"""
        import numpy as np
        
        if hasattr(values, 'tolist'):
            values_list = values.tolist()
        else:
            values_list = list(values)
            
        # Округляем значения для отображения
        if 'gyro' in param_id or 'attitude' in param_id:
            display_values = [round(v, 2) for v in values_list]
        else:
            display_values = values_list
            
        return {
            'id': param_id,
            'name': name,
            'unit': unit,
            'icon': icon,
            'color': color,
            'values': values_list
        }
    
    def get_chart_data(self, param_id):
        """Данные для графика"""
        try:
            import numpy as np
            
            # Пробуем разобрать param_id
            if param_id == 'gps_alt':
                topic, field = 'vehicle_gps_position', 'alt'
                conversion = 'mm_to_m'
            elif param_id == 'gps_speed':
                topic, field = 'vehicle_gps_position', 'vel_m_s'
                conversion = 'mps_to_kmh'
            elif param_id == 'gps_satellites':
                topic, field = 'vehicle_gps_position', 'satellites_used'
                conversion = None
            elif param_id == 'gps_fix_type':
                topic, field = 'vehicle_gps_position', 'fix_type'
                conversion = None
            elif param_id == 'global_alt':
                topic, field = 'vehicle_global_position', 'alt'
                conversion = None
            elif param_id == 'global_speed':
                topic, field = 'vehicle_global_position', 'vel_n'
                conversion = 'vector_speed'
            elif param_id == 'local_z':
                topic, field = 'vehicle_local_position', 'z'
                conversion = 'invert'
            elif param_id.startswith('local_v'):
                topic, field = 'vehicle_local_position', param_id.replace('local_', '')
                conversion = None
            elif param_id == 'local_distance':
                topic, field = 'vehicle_local_position', 'x'
                conversion = 'local_distance'
            elif param_id.startswith('attitude_'):
                topic, field = 'vehicle_attitude', param_id.replace('attitude_', '')
                conversion = 'rad_to_deg'
            elif param_id.startswith('battery_'):
                topic, field = 'battery_status', param_id.replace('battery_', '')
                conversion = 'percent' if 'remaining' in param_id else None
            elif param_id.startswith('accel_'):
                axis = param_id.replace('accel_', '')
                topic, field = 'sensor_combined', f'accelerometer_m_s2[{["x","y","z"].index(axis)}]'
                conversion = None
            elif param_id.startswith('gyro_'):
                axis = param_id.replace('gyro_', '')
                topic, field = 'sensor_combined', f'gyro_rad[{["x","y","z"].index(axis)}]'
                conversion = 'rad_to_deg'
            elif param_id.startswith('motor_'):
                motor_num = int(param_id.replace('motor_', '')) - 1
                topic, field = 'actuator_outputs', f'output[{motor_num}]'
                conversion = None
            elif param_id.startswith('rc_'):
                chan_num = int(param_id.replace('rc_', '')) - 1
                topic, field = 'rc_channels', f'channels[{chan_num}]'
                conversion = None
            else:
                # Для пользовательских параметров
                parts = param_id.split('_', 1)
                if len(parts) == 2:
                    topic, field = parts[0], parts[1]
                    conversion = None
                else:
                    return None
            
            if topic not in self.topics_data:
                return None
            
            df = self.topics_data[topic]
            if field not in df.columns:
                return None
            
            # Подготавливаем данные
            x = df['time_sec'].tolist() if 'time_sec' in df.columns else list(range(len(df)))
            y = df[field].dropna().tolist()
            
            # Применяем преобразование
            if conversion == 'mps_to_kmh':
                y = [v * 3.6 for v in y]
            elif conversion == 'mm_to_m':
                y = [v / 1000 for v in y]  # мм → метры
            elif conversion == 'rad_to_deg':
                y = [math.degrees(v) for v in y]
            elif conversion == 'invert':
                y = [-v for v in y]  # Инвертируем высоту
            elif conversion == 'percent' and 'remaining' in field:
                y = [v * 100 for v in y]
            elif conversion == 'vector_speed':
                # Для глобальной скорости нужны оба компонента
                if 'vel_e' in df.columns:
                    vel_n = df[field].dropna().tolist()
                    vel_e = df['vel_e'].dropna().tolist()
                    # Вычисляем горизонтальную скорость
                    y = [np.sqrt(vn**2 + ve**2) for vn, ve in zip(vel_n, vel_e)]
            elif conversion == 'local_distance':
                # Для локальной дистанции нужны X и Y
                if 'y' in df.columns:
                    x_vals = df[field].dropna().tolist()
                    y_vals = df['y'].dropna().tolist()
                    # Вычисляем дистанцию от начала
                    y = [np.sqrt(xv**2 + yv**2) for xv, yv in zip(x_vals, y_vals)]
            
            return {'x': x, 'y': y}
        
        except Exception as e:
            print(f"Ошибка получения данных для {param_id}: {e}")
            return None

# ========== ГЛАВНАЯ СТРАНИЦА ==========
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PX4 Log Analyzer PRO</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 500px;
                width: 100%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                text-align: center;
            }
            .logo { font-size: 48px; margin-bottom: 20px; }
            h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
            .subtitle { color: #666; margin-bottom: 30px; line-height: 1.5; }
            .upload-area {
                border: 3px dashed #ddd;
                border-radius: 12px;
                padding: 50px 20px;
                margin: 30px 0;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover { border-color: #667eea; background: #f8f9ff; }
            .upload-icon { font-size: 60px; margin-bottom: 20px; opacity: 0.7; }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 16px 40px;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-top: 20px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            }
            .file-input { display: none; }
            .note {
                color: #888;
                font-size: 14px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }
            .features {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 20px;
                text-align: left;
                font-size: 14px;
            }
            .feature {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🚁</div>
            <h1>PX4 Log Analyzer PRO</h1>
            <p class="subtitle">Расширенный анализ телеметрии полетов</p>
            
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <h3>Загрузите .ulg файл</h3>
                   
                </div>
                
                <input type="file" id="fileInput" name="file" accept=".ulg" class="file-input" onchange="this.form.submit()" required>
                
               
                <div class="note">
                    📁 Поддерживаются файлы .ulg от автопилота PX4
                </div>
            </form>
        </div>
    </body>
    </html>
    """

# ========== ЗАГРУЗКА ==========
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Файл не выбран", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "Файл не выбран", 400
    
    if not file.filename.endswith('.ulg'):
        return "Только .ulg файлы", 400
    
    file_id = str(uuid.uuid4())[:8]
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.ulg")
    file.save(temp_path)
    
    try:
        analyzer = EnhancedAnalyzer(temp_path)
        result = analyzer.analyze()
        
        if not result['success']:
            os.remove(temp_path)
            return f"Ошибка анализа: {result['error']}", 500
        
        # Получаем параметры
        params_info = analyzer.get_best_parameters()
        
        # Сохраняем
        session_data = {
            'file_id': file_id,
            'filename': file.filename,
            'duration': result['duration'],
            'start_time': result['start_time'].strftime('%H:%M:%S'),
            'topics_count': result['topics_count'],
            'has_gps': result['has_gps'],
            'gps_points': len(analyzer.gps_coords),
            'parameters': params_info['all'],
            'categories': params_info['categories'],
            'file_path': temp_path
        }
        
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # Очищаем временный файл если он большой
        if os.path.getsize(temp_path) > 50 * 1024 * 1024:  # 50 MB
            os.remove(temp_path)
            session_data['file_path'] = None
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/dashboard/{file_id}">
        </head>
        <body>
            <p>Анализ завершен. Найдено {params_info['total_count']} параметров, 
            GPS точек: {len(analyzer.gps_coords)}.</p>
        </body>
        </html>
        """
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"Ошибка: {str(e)}", 500

# ========== ДАШБОРД ==========
@app.route('/dashboard/<file_id>')
def dashboard(file_id):
    session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    
    if not os.path.exists(session_path):
        return """
        <div style="text-align: center; padding: 40px;">
            <h2>❌ Файл не найден</h2>
            <p>Сессия истекла или файл удален</p>
            <a href="/" style="
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                text-decoration: none;
                margin-top: 20px;
            ">← На главную</a>
        </div>
        """, 404
    
    with open(session_path, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    
    # Генерация HTML по категориям
    categories_html = ""
    for category_name, params in session_data['categories'].items():
        cards_html = ""
        for param in params:
            cards_html += f"""
            <div class="card" onclick="loadChart('{param['id']}')">
                <div class="card-icon">{param['icon']}</div>
                <div class="card-title">{param['name']}</div>
                <div class="card-value">{param['stats']['current']:.1f}</div>
                <div class="card-unit">{param['unit']}</div>
                <div class="card-range">
                    {param['stats']['min']:.1f} – {param['stats']['max']:.1f}
                </div>
            </div>
            """
        
        categories_html += f"""
        <div class="category-section">
            <div class="category-title">{category_name} ({len(params)})</div>
            <div class="cards-grid">
                {cards_html}
            </div>
        </div>
        """
    
    # Кнопки выбора для всех параметров
    buttons_html = ""
    for param in session_data['parameters'][:15]:  # Первые 15 параметров
        buttons_html += f"""
        <button class="chart-btn" onclick="loadChart('{param['id']}')">
            {param['icon']} {param['name']}
        </button>
        """
    
    # HTML для карты если есть GPS
# HTML для карты если есть GPS
map_html = ""
if session_data['has_gps'] and session_data['gps_points'] > 0:
    map_html = f"""
<div class="section">
<div class="section-title">
    🗺️ Траектория полета ({session_data['gps_points']} точек)
    <button onclick="showGpsTable()" style="
        margin-left: 15px;
        padding: 6px 12px;
        background: #27ae60;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 13px;
        cursor: pointer;
    ">📋 Все координаты</button>
</div>
<div id="map" style="height: 400px; border-radius: 12px; margin-top: 20px;"></div>
<div style="margin-top: 10px; font-size: 14px; color: #666;">
    Для навигации используйте колесо мыши, для перемещения — зажатие левой кнопки
</div>

<!-- Модальное окно -->
<div id="gpsModal" style="
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0; top: 0;
    width: 100%; height: 100%;
    background-color: rgba(0,0,0,0.6);
">
    <div style="
        background: white;
        margin: 40px auto;
        padding: 20px;
        width: 90%;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    ">
        <h3 style="margin-bottom: 15px;">📍 Все GPS-точки полёта</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: #f1f1f1;">
                    <th style="padding: 8px; text-align: left;">№</th>
                    <th style="padding: 8px; text-align: left;">Широта</th>
                    <th style="padding: 8px; text-align: left;">Долгота</th>
                    <th style="padding: 8px; text-align: left;">Высота (м)</th>
                </tr>
            </thead>
            <tbody id="gpsTableBody">
            </tbody>
        </table>
        <button onclick="closeGpsModal()" style="
            margin-top: 15px;
            padding: 8px 20px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        ">Закрыть</button>
    </div>
</div>
</div>
"""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Анализ: {session_data['filename']}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8f9fa;
                color: #333;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .header h1 {{
                font-size: 22px;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            
            .stats-badge {{
                background: #2ecc71;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }}
            
            .gps-badge {{
                background: #3498db;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }}
            
            .file-info {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                color: #666;
                font-size: 14px;
            }}
            
            .file-info span {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            
            .category-section {{
                margin-bottom: 25px;
            }}
            
            .category-title {{
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #3498db;
            }}
            
            .cards-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                gap: 15px;
                margin-bottom: 10px;
            }}
            
            .card {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.08);
                cursor: pointer;
                transition: all 0.3s;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            }}
            
            .card-icon {{
                font-size: 28px;
                margin-bottom: 10px;
            }}
            
            .card-title {{
                font-size: 16px;
                color: #555;
                margin-bottom: 8px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            
            .card-value {{
                font-size: 32px;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 4px;
            }}
            
            .card-unit {{
                font-size: 14px;
                color: #7f8c8d;
                margin-bottom: 8px;
            }}
            
            .card-range {{
                font-size: 12px;
                color: #95a5a6;
                padding-top: 8px;
                border-top: 1px solid #eee;
            }}
            
            .section {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.08);
            }}
            
            .section-title {{
                font-size: 20px;
                margin-bottom: 20px;
                color: #2c3e50;
            }}
            
            .chart-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }}
            
            .chart-btn {{
                padding: 10px 16px;
                background: white;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 14px;
                white-space: nowrap;
            }}
            
            .chart-btn.active {{
                background: #3498db;
                color: white;
                border-color: #3498db;
            }}
            
            .chart-btn:hover {{
                border-color: #3498db;
            }}
            
            .chart-container {{
                height: 450px;
                width: 100%;
                border-radius: 8px;
                overflow: hidden;
            }}
            
            .actions {{
                display: flex;
                gap: 15px;
                margin-top: 25px;
            }}
            
            .action-btn {{
                padding: 12px 24px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                font-size: 15px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            
            .action-btn:hover {{
                background: #2980b9;
            }}
            
            .action-btn.secondary {{
                background: #95a5a6;
            }}
            
            .action-btn.secondary:hover {{
                background: #7f8c8d;
            }}
            
            @media (max-width: 768px) {{
                .cards-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
                .chart-buttons {{
                    overflow-x: auto;
                    padding-bottom: 10px;
                }}
            }}
            
            @media (max-width: 480px) {{
                .cards-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>
                    🚁 Анализ полета: {session_data['filename']}
                    <span class="stats-badge">{len(session_data['categories'])} категорий, {len(session_data['parameters'])} параметров</span>
                    {f'<span class="gps-badge">📍 {session_data["gps_points"]} GPS точек</span>' if session_data['has_gps'] else ''}
                </h1>
                <div class="file-info">
                    <span>⏱️ Длительность: {session_data['duration']:.1f} сек</span>
                    <span>📊 Топиков: {session_data['topics_count']}</span>
                    <span>📍 GPS: {'Да (' + str(session_data['gps_points']) + ' точек)' if session_data['has_gps'] else 'Нет'}</span>
                </div>
            </div>
            
            {categories_html}
            
            <div class="section">
                <div class="section-title">📈 Детальный график</div>
                
                <div class="chart-buttons" id="chartButtons">
                    {buttons_html}
                </div>
                
                <div class="chart-container" id="chart">
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%; color: #7f8c8d;">
                        Кликните на параметр выше для отображения графика
                    </div>
                </div>
            </div>
            
            {map_html}
            
            <div class="actions">
                <a href="/" class="action-btn secondary">
                    ← Новый анализ
                </a>
                <button onclick="exportData()" class="action-btn">
                    📥 Экспорт данных
                </button>
                <button onclick="showAllParams()" class="action-btn" style="background: #9b59b6;">
                    📋 Все параметры
                </button>
                <button onclick="downloadKML()" class="action-btn" style="background: #27ae60;" {'' if session_data['has_gps'] and session_data['gps_points'] > 0 else 'disabled'}>
                    🗺️ Скачать KML
                </button>
            </div>
        </div>
        
        <script>
            const fileId = '{file_id}';
            const parameters = {json.dumps(session_data['parameters'], ensure_ascii=False)};
            const hasGPS = {json.dumps(session_data['has_gps'])};
            const gpsPoints = {json.dumps(session_data['gps_points'])};
            
            async function loadChart(paramId) {{
                // Обновляем активную кнопку
                document.querySelectorAll('.chart-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                const chartDiv = document.getElementById('chart');
                chartDiv.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100%;">Загрузка графика...</div>';
                
                try {{
                    const response = await fetch(`/api/chart/${{fileId}}/${{paramId}}`);
                    const data = await response.json();
                    
                    if (data.error) {{
                        chartDiv.innerHTML = `<div style="text-align: center; padding: 40px; color: #e74c3c;">${{data.error}}</div>`;
                        return;
                    }}
                    
                    Plotly.newPlot('chart', data.data, data.layout, {{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }});
                    
                }} catch (error) {{
                    chartDiv.innerHTML = `<div style="text-align: center; padding: 40px; color: #e74c3c;">Ошибка: ${{error.message}}</div>`;
                }}
            }}
            
            function exportData() {{
                window.open(`/api/export/${{fileId}}`, '_blank');
            }}
            
            function showAllParams() {{
                const paramList = parameters.map(p => `• ${{p.icon}} ${{p.name}}: ${{p.stats.current.toFixed(2)}} ${{p.unit}}`).join('\\n');
                alert(`Все параметры (всего ${{parameters.length}}):\\n\\n${{paramList}}`);
            }}
            
            function downloadKML() {{
                window.open(`/api/kml/${{fileId}}`, '_blank');
            }}

            function showGpsTable() {
    fetch(`/api/gps/${fileId}`)
        .then(response => response.json())
        .then(coords => {
            const tbody = document.getElementById('gpsTableBody');
            tbody.innerHTML = '';
            coords.forEach((coord, i) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${coord[0].toFixed(6)}</td>
                    <td>${coord[1].toFixed(6)}</td>
                    <td>${parseFloat(coord[2]).toFixed(1)}</td>
                `;
                tbody.appendChild(row);
            });
            document.getElementById('gpsModal').style.display = 'block';
        });
}

function closeGpsModal() {
    document.getElementById('gpsModal').style.display = 'none';
}

// Закрытие по клику вне окна
document.addEventListener('click', function(e) {
    const modal = document.getElementById('gpsModal');
    if (modal.style.display === 'block' && e.target === modal) {
        closeGpsModal();
    }
});
        
            // Инициализация карты если есть GPS
            if (hasGPS && gpsPoints > 0) {{
                setTimeout(() => {{
                    fetch(`/api/gps/${{fileId}}`)
                        .then(response => response.json())
                        .then(coords => {{
                            if (coords.length > 0) {{
                                // Вычисляем центр траектории
                                const lats = coords.map(c => c[0]);
                                const lons = coords.map(c => c[1]);
                                const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
                                const centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;
                                
                                const map = L.map('map', {{
                                    attributionControl: false
                                }}).setView([centerLat, centerLon], 15);
                                
                                L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                                    maxZoom: 20,
                                    attribution: false
                                }}).addTo(map);
                               
                                
const points = coords.map(c => [c[0], c[1]]);

// Линия траектории
const track = L.polyline(points, {
    color: '#3498db',
    weight: 3,
    opacity: 0.8,
    smoothFactor: 1
}).addTo(map);

// Добавляем невидимые маркеры для подсказок
const markerGroup = L.layerGroup().addTo(map);
coords.forEach((coord, i) => {
    const lat = coord[0];
    const lon = coord[1];
    const alt = coord[2].toFixed(1);
    const marker = L.circleMarker([lat, lon], {
        radius: 0,
        fillOpacity: 0,
        stroke: false
    }).on('mouseover', function(e) {
        const popup = L.popup()
            .setLatLng([lat, lon])
            .setContent(`
                <div style="font-family: monospace; font-size: 13px;">
                    Точка ${i + 1}<br>
                    Широта: ${lat.toFixed(6)}°<br>
                    Долгота: ${lon.toFixed(6)}°<br>
                    Высота: ${alt} м
                </div>
            `)
            .openOn(map);
    }).on('mouseout', function() {
        map.closePopup();
    });
    markerGroup.addLayer(marker);
});
                                
                                // Добавляем маркеры взлета и посадки
                                if (points.length > 0) {{
                                    L.marker(points[0], {{
                                        icon: L.divIcon({{
                                            html: '🚀',
                                            className: 'flight-marker',
                                            iconSize: [30, 30]
                                        }})
                                    }}).addTo(map).bindPopup('Взлет');
                                    
                                    L.marker(points[points.length-1], {{
                                        icon: L.divIcon({{
                                            html: '🛬',
                                            className: 'flight-marker',
                                            iconSize: [30, 30]
                                        }})
                                    }}).addTo(map).bindPopup('Посадка');
                                    
                                    // Масштабируем чтобы вся траектория была видна
                                    map.fitBounds(track.getBounds());
                                    
                                    // Добавляем информацию о маршруте
                                    const info = L.control({{position: 'topright'}});
                                    info.onAdd = function() {{
                                        const div = L.DomUtil.create('div', 'map-info');
                                        div.innerHTML = `
                                            <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 12px;">
                                                <strong>Маршрут</strong><br>
                                                Точки: ${{coords.length}}<br>
                                                Длина: ~${{calculateDistance(points).toFixed(2)}} км
                                            </div>
                                        `;
                                        return div;
                                    }};
                                    info.addTo(map);
                                }}
                            }}
                        }})
                        .catch(error => {{
                            console.error('Ошибка загрузки GPS данных:', error);
                            document.getElementById('map').innerHTML = `
                                <div style="display: flex; justify-content: center; align-items: center; height: 100%; color: #e74c3c;">
                                    Ошибка загрузки GPS данных
                                </div>
                            `;
                        }});
                    
                    // Функция для расчета расстояния
                    function calculateDistance(points) {{
                        let totalDistance = 0;
                        for (let i = 1; i < points.length; i++) {{
                            const lat1 = points[i-1][0];
                            const lon1 = points[i-1][1];
                            const lat2 = points[i][0];
                            const lon2 = points[i][1];
                            
                            // Формула гаверсинусов для расчета расстояния на сфере
                            const R = 6371; // Радиус Земли в км
                            const dLat = (lat2 - lat1) * Math.PI / 180;
                            const dLon = (lon2 - lon1) * Math.PI / 180;
                            const a = 
                                Math.sin(dLat/2) * Math.sin(dLat/2) +
                                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
                                Math.sin(dLon/2) * Math.sin(dLon/2);
                            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                            totalDistance += R * c;
                        }}
                        return totalDistance;
                    }}
                }}, 500);
            }}
            
            // Автоматически загружаем первый график
            document.addEventListener('DOMContentLoaded', function() {{
                if (parameters.length > 0) {{
                    const firstBtn = document.querySelector('.chart-btn');
                    if (firstBtn) {{
                        firstBtn.click();
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

# ========== API ГРАФИКОВ ==========
@app.route('/api/chart/<file_id>/<param_id>')
def get_chart(file_id, param_id):
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Находим параметр
        param = None
        for p in session_data['parameters']:
            if p['id'] == param_id:
                param = p
                break
        
        if not param:
            return jsonify({'error': 'Параметр не найден'}), 404
        
        # Получаем данные если файл еще существует
        if session_data.get('file_path') and os.path.exists(session_data['file_path']):
            analyzer = EnhancedAnalyzer(session_data['file_path'])
            analyzer.analyze()
        else:
            # Если файл удален, создаем пустой анализатор
            analyzer = EnhancedAnalyzer(None)
            analyzer.topics_data = {}
        
        chart_data = analyzer.get_chart_data(param_id)
        if not chart_data or len(chart_data['x']) == 0:
            return jsonify({'error': 'Нет данных для графика'}), 404
        
        # Создаем график
        fig = go.Figure()
        
        # Цвет с альфа-каналом
        color = param['color']
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        fig.add_trace(go.Scatter(
            x=chart_data['x'],
            y=chart_data['y'],
            mode='lines',
            name=param['name'],
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
            hovertemplate=(
                f'<b>{param["name"]}</b><br>' +
                'Время: %{x:.1f} сек<br>' +
                f'Значение: %{{y:.2f}} {param["unit"]}<br>' +
                '<extra></extra>'
            )
        ))
        
        # Настройки
        fig.update_layout(
            title=dict(
                text=f'{param["icon"]} {param["name"]}',
                font=dict(size=18, color='#2c3e50'),
                x=0.5
            ),
            xaxis=dict(
                title='Время полета (сек)',
                gridcolor='#f0f0f0',
                linecolor='#ddd'
            ),
            yaxis=dict(
                title=f'{param["name"]} ({param["unit"]})',
                gridcolor='#f0f0f0',
                linecolor='#ddd'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(bgcolor='white', font_size=14),
            margin=dict(l=50, r=30, t=50, b=50),
            height=400,
            showlegend=False
        )
        
        return jsonify({
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== API GPS КООРДИНАТ ==========
@app.route('/api/gps/<file_id>')
def get_gps_coords(file_id):
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Если файл существует, перезагружаем данные
        if session_data.get('file_path') and os.path.exists(session_data['file_path']):
            analyzer = EnhancedAnalyzer(session_data['file_path'])
            analyzer.analyze()
            return jsonify(analyzer.gps_coords)
        else:
            # Если файл удален, возвращаем пустой массив
            return jsonify([])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== API KML ЭКСПОРТ ==========
@app.route('/api/kml/<file_id>')
def get_kml(file_id):
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        if not session_data['has_gps'] or session_data['gps_points'] == 0:
            return "Нет GPS данных для экспорта", 404
        
        # Получаем GPS координаты
        analyzer = EnhancedAnalyzer(session_data['file_path'])
        analyzer.analyze()
        
        if not analyzer.gps_coords:
            return "Нет GPS данных", 404
        
        # Создаем KML файл
        kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Flight Track - {session_data['filename']}</name>
    <description>Flight track exported from PX4 Log Analyzer</description>
    <Style id="trackStyle">
      <LineStyle>
        <color>ff3498db</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#trackStyle</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
'''
        
        # Добавляем координаты
        for coord in analyzer.gps_coords:
            kml_content += f'          {coord[1]},{coord[0]},{coord[2]}\n'
        
        kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Takeoff</name>
      <Point>
        <coordinates>
'''
        if analyzer.gps_coords:
            kml_content += f'          {analyzer.gps_coords[0][1]},{analyzer.gps_coords[0][0]},{analyzer.gps_coords[0][2]}\n'
        
        kml_content += '''        </coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>Landing</name>
      <Point>
        <coordinates>
'''
        if analyzer.gps_coords:
            kml_content += f'          {analyzer.gps_coords[-1][1]},{analyzer.gps_coords[-1][0]},{analyzer.gps_coords[-1][2]}\n'
        
        kml_content += '''        </coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>'''
        
        from io import BytesIO
        buffer = BytesIO(kml_content.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/vnd.google-earth.kml+xml',
            as_attachment=True,
            download_name=f'flight_track_{file_id}.kml'
        )
        
    except Exception as e:
        return f"Ошибка экспорта KML: {str(e)}", 500

@app.route('/api/export/<file_id>')
def export_data(file_id):
    """Экспорт в CSV с правильной кодировкой для Windows"""
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
        
        if not os.path.exists(session_path):
            return "Сессия не найдена", 404
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # 1. Создаем CSV с разделителем ";" (европейский стандарт)
        csv_lines = []
        
        # 2. Общая информация
        csv_lines.append("Отчет анализа лога PX4")
        csv_lines.append("")
        csv_lines.append(f"Файл;{session_data['filename']}")
        csv_lines.append(f"Длительность;{session_data['duration']:.1f} сек")
        csv_lines.append(f"Дата;{session_data['date']}")
        csv_lines.append(f"Время;{session_data['start_time']}")
        csv_lines.append(f"Всего параметров;{len(session_data['parameters'])}")
        csv_lines.append(f"GPS точек;{session_data['gps_points']}")
        csv_lines.append("")
        
        # 3. Заголовки таблицы
        csv_lines.append("Категория;Параметр;Текущее;Минимум;Максимум;Среднее;Единицы")
        
        # 4. Данные - убираем эмодзи из категорий и названий
        def remove_emojis(text):
            """Удаляет эмодзи и специальные символы"""
            import re
            # Паттерн для удаления эмодзи
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # эмотиконы
                u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
                u"\U0001F680-\U0001F6FF"  # транспорт и символы карт
                u"\U0001F1E0-\U0001F1FF"  # флаги (iOS)
                "]+", flags=re.UNICODE)
            
            # Удаляем эмодзи и лишние пробелы
            cleaned = emoji_pattern.sub(r'', text)
            return cleaned.strip()
        
        for category_name, params in session_data['categories'].items():
            # Убираем эмодзи из категории
            clean_category = remove_emojis(category_name)
            
            for param in params:
                # Убираем эмодзи из названия параметра
                clean_name = remove_emojis(param["name"])
                
                line = (
                    f'{clean_category};'
                    f'{clean_name};'
                    f'{param["stats"]["current"]:.2f};'
                    f'{param["stats"]["min"]:.2f};'
                    f'{param["stats"]["max"]:.2f};'
                    f'{param["stats"]["avg"]:.2f};'
                    f'{param["unit"]}'
                )
                csv_lines.append(line)
        
        # 5. КОНВЕРТИРУЕМ В ПРАВИЛЬНУЮ КОДИРОВКУ
        csv_content = "\n".join(csv_lines)
        
        # Вариант 1: Windows-1251 (кириллица для Windows)
        try:
            csv_bytes = csv_content.encode('windows-1251')
        except UnicodeEncodeError:
            # Если есть символы, которые нельзя закодировать в 1251
            csv_bytes = csv_content.encode('utf-8-sig')  # UTF-8 с BOM
        
        from io import BytesIO
        buffer = BytesIO(csv_bytes)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='text/csv; charset=windows-1251',
            as_attachment=True,
            download_name=f'analysis_{file_id}.csv'
        )
        
    except Exception as e:
        return f"Ошибка экспорта: {str(e)}", 500

# ========== ЗАПУСК ==========
if __name__ == '__main__':
    print("=" * 60)
    print("PX4 Log Analyzer PRO - Расширенная версия")
    print("• 20+ параметров с группировкой по категориям")
    print("• Исправленная высота (инвертированная для локальной)")
    print("• Поддержка GPS карт и экспорт KML")
    print("=" * 60)
    print("Сервер запущен: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
