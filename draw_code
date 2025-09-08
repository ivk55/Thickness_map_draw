import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import io
import base64
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import matplotlib.font_manager as fm

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EquipmentConfig:
    def __init__(self):
        self.equipment_types = {
            'PHDP': {
                'has_sides': False,
                'angle_fixed': True,
                'fixed_angle': 0,
                'supports_thk': True,
                'max_thk_count': 24,
                'supports_nozzle': True,  # 노즐 맵 지원
                'units': ['TAHP307', 'TAHP701', 'TAHP702', 'TAHP703', 'TAHP851', 'TAHP852', 
                         'TAHP853', 'TAHP854', 'TAHP855', 'TAHP856', 'TAHP857', 'TAJP7U1', 
                         'TAJP7U2', 'TAJP7U3']
            },
            'SIOC': {
                'has_sides': True,
                'angle_fixed': False,
                'default_angles': {
                    'LP1': {'side1': 336, 'side2': 180},
                    'LP2': {'side1': 336, 'side2': 180},
                    'LP3': {'side1': 180, 'side2': 26}
                },
                'supports_thk': False,
                'units': ['TAKP851', 'TAKP852', 'TAKP861', 'TAKP863', 'TAKP864', 'TAKP865', 'TAKP866', 'TAKP867', 'TAKP853']
            },
            'TEOS': {
                'has_sides': True,
                'angle_fixed': False,
                'default_angles': {
                    'LP1': {'side1': 336, 'side2': 180},
                    'LP2': {'side1': 336, 'side2': 180},
                    'LP3': {'side1': 180, 'side2': 26}
                },
                'supports_thk': False,
                'side_names': {'side1': 'Side1', 'side2': 'Side2'},
                'supports_exclude_point20': True,  # 20번 포인트 제외 옵션 지원
                'units': ['TATP871', 'TATP872', 'TATP873', 'TATP874']
            },
            'MIR3000': {
                'has_sides': False,
                'angle_fixed': True,
                'fixed_angle': 0,
                'supports_thk': True,
                'max_thk_count': 24,
                'units': ['TBLP3U4', 'TBLP3U5', 'TBLP7U1', 'TBLP7U2', 'TBLP7U3', 'TBLP7U4', 'TBLP7U5']
            },
            'TES': {
                'has_sides': False,
                'angle_fixed': True,
                'fixed_angle': 0,
                'supports_thk': False,
                'units': ['TEXP7U1', 'TEAP360']
            },
            'TEAP360': {
                'has_sides': False,
                'angle_fixed': True,
                'fixed_angle': 0,
                'supports_thk': True,
                'max_thk_count': 24,
                'units': ['TEAP360']
            }
        }
        
        self.new_efem_units = ['TAKP867', 'TAKP853', 'TATP874']
        self.new_efem_angles = {
            'LP1': {'side1': 336, 'side2': 220},
            'LP2': {'side1': 336, 'side2': 220},
            'LP3': {'side1': 140, 'side2': 24}
        }
        
        # TEAP360 특별 처리 (TES에 속하지만 THK 지원)
        self.teap360_special_units = ['TEAP360']
    
    def is_new_efem(self, unit):
        return unit in self.new_efem_units
    
    def is_teap360_special(self, unit):
        return unit in self.teap360_special_units
    
    def get_default_angles(self, equipment_type, unit, lp):
        if equipment_type in ['SIOC', 'TEOS']:
            if self.is_new_efem(unit):
                return self.new_efem_angles[lp]
            else:
                return self.equipment_types[equipment_type]['default_angles'][lp]
        return {'side1': 0, 'side2': 0}

class WaferMapping:
    def __init__(self):
        self.coordinates = np.array([
            [1, 0, 0], [2, 0, 49], [3, -49, 0], [4, 0, -49], [5, 49, 0],
            [6, 0, 98], [7, -69.3, 69.3], [8, -98, 0], [9, -69.3, -69.3], [10, 0, -98],
            [11, 69.3, -69.3], [12, 98, 0], [13, 69.3, 69.3], [14, 0, 147], [15, -73.5, 127.31],
            [16, -127.31, 73.5], [17, -147.0, 0], [18, -127.31, -73.5], [19, -73.5, -127.31], [20, 0, -147],
            [21, 73.5, -127.31], [22, 127.31, -73.5], [23, 147, 0], [24, 127.31, 73.5], [25, 73.5, 127.31]
        ])
        
        self.wafer_radius = 150
        self.nozzle_radius = 200  # 노즐 반지름
        self.grid_resolution = 1
        self.interpolation_method = 'cubic'
        self.colormap = 'jet'
        
        # 노즐 좌표 생성 (PHDP용) - 다른 변수들 정의 후에 호출
        self.nozzle_coordinates = self.create_nozzle_coordinates()

    def create_nozzle_coordinates(self):
        """32개 노즐 좌표 생성 (12시 32번부터 반시계방향)"""
        nozzle_coords = []
        radius = self.nozzle_radius
        
        # 12시(90도)부터 시작해서 반시계방향으로 32, 31, 30, ..., 1
        for i in range(32):
            nozzle_num = 32 - i  # 32부터 1까지
            
            # 각도 계산 (12시부터 반시계방향)
            angle = 90 + (i * 360 / 32)  # 90도부터 시작해서 반시계방향
            angle_rad = np.deg2rad(angle)
            
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            
            nozzle_coords.append([nozzle_num, x, y])
            
        return np.array(nozzle_coords)

    def create_wafer_map(self, side1_angle, side2_angle, side1_data, side2_data, exclude_point20=False):
        n_points = len(self.coordinates)
        data = np.zeros((n_points, 5))
        data[:, 0:3] = self.coordinates
        data[:, 3] = side1_data
        data[:, 4] = side2_data

        # 원본 좌표
        X = data[:, 1]
        Y = data[:, 2]
        Z = data[:, 3]
        Z1 = data[:, 4]

        # Side1 회전 매트릭스
        side1_rad = np.deg2rad(side1_angle)
        R1 = np.array([
            [np.cos(side1_rad), -np.sin(side1_rad)],
            [np.sin(side1_rad), np.cos(side1_rad)]
        ])
        
        # Side2 회전 매트릭스
        side2_rad = np.deg2rad(side2_angle)
        R2 = np.array([
            [np.cos(side2_rad), -np.sin(side2_rad)],
            [np.sin(side2_rad), np.cos(side2_rad)]
        ])

        # Side1 회전 적용
        XY_side1 = R1 @ np.vstack([X, Y])
        X_rot_side1 = XY_side1[0, :]
        Y_rot_side1 = XY_side1[1, :]
        
        # Side2 회전 적용
        XY_side2 = R2 @ np.vstack([X, Y])
        X_rot_side2 = XY_side2[0, :]
        Y_rot_side2 = XY_side2[1, :]

        # 그리드 생성
        x_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        y_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        xq, yq = np.meshgrid(x_range, y_range)

        # 보간
        zq_side1 = griddata((X_rot_side1, Y_rot_side1), Z, (xq, yq), method=self.interpolation_method)
        zq_side2 = griddata((X_rot_side2, Y_rot_side2), Z1, (xq, yq), method=self.interpolation_method)

        # 웨이퍼 영역 마스크
        mask = np.sqrt(xq ** 2 + yq ** 2) > self.wafer_radius
        zq_side1[mask] = np.nan
        zq_side2[mask] = np.nan

        return {
            'data': data,
            'X_rot_side1': X_rot_side1, 'Y_rot_side1': Y_rot_side1,
            'X_rot_side2': X_rot_side2, 'Y_rot_side2': Y_rot_side2,
            'xq': xq, 'yq': yq,
            'zq_side1': zq_side1, 'zq_side2': zq_side2,
            'side1_angle': side1_angle, 'side2_angle': side2_angle,
            'Z': Z, 'Z1': Z1,
            'exclude_point20': exclude_point20
        }

    def create_thk_map(self, thk_datasets, angle=0, show_nozzle=False):
        n_points = len(self.coordinates)
        X = self.coordinates[:, 1]
        Y = self.coordinates[:, 2]

        # 회전 매트릭스
        angle_rad = np.deg2rad(angle)
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # 회전 적용
        XY_rot = R @ np.vstack([X, Y])
        X_rot = XY_rot[0, :]
        Y_rot = XY_rot[1, :]

        # 그리드 생성
        x_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        y_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        xq, yq = np.meshgrid(x_range, y_range)

        # 웨이퍼 영역 마스크
        mask = np.sqrt(xq ** 2 + yq ** 2) > self.wafer_radius

        result = {
            'X_rot': X_rot, 'Y_rot': Y_rot,
            'xq': xq, 'yq': yq,
            'angle': angle,
            'thk_maps': [],
            'thk_data': thk_datasets,
            'show_nozzle': show_nozzle
        }

        # 각 THK 데이터셋에 대해 보간 수행
        for i, thk_data in enumerate(thk_datasets):
            zq_thk = griddata((X_rot, Y_rot), thk_data, (xq, yq), method=self.interpolation_method)
            zq_thk[mask] = np.nan
            result['thk_maps'].append(zq_thk)

        return result

    def plot_thk_map_plotly(self, result, selected_datasets=None):
        from plotly.subplots import make_subplots
        import math
        
        if selected_datasets is None:
            selected_datasets = list(range(len(result['thk_data'])))
        
        num_datasets = len(selected_datasets)
        if num_datasets == 0:
            return None
            
        # 그리드 계산 (최대 4x6 = 24개)
        if num_datasets == 1:
            rows, cols = 1, 1
        elif num_datasets == 2:
            rows, cols = 1, 2
        elif num_datasets <= 4:
            rows, cols = 2, 2
        elif num_datasets <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 6
        
        # 서브플롯 제목 생성
        subplot_titles = [f'THK Data {i+1}' for i in selected_datasets]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # 웨이퍼 경계선 좌표
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.wafer_radius * np.cos(theta)
        circle_y = self.wafer_radius * np.sin(theta)
        
        # 노즐 좌표 (PHDP용)
        if result.get('show_nozzle', False):
            nozzle_x = self.nozzle_coordinates[:, 1]
            nozzle_y = self.nozzle_coordinates[:, 2]
            nozzle_nums = self.nozzle_coordinates[:, 0].astype(int)
        
        for idx, dataset_idx in enumerate(selected_datasets):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # 히트맵 추가
            fig.add_trace(
                go.Heatmap(
                    z=result['thk_maps'][dataset_idx],
                    x=result['xq'][0, :],
                    y=result['yq'][:, 0],
                    colorscale='Jet',
                    showscale=(idx == 0),  # 첫 번째만 컬러바 표시
                    colorbar=dict(title="Thickness (nm)", x=1.02) if idx == 0 else None,
                    name=f"THK {dataset_idx+1}"
                ),
                row=row, col=col
            )
            
            # 측정점 추가
            fig.add_trace(
                go.Scatter(
                    x=result['X_rot'],
                    y=result['Y_rot'],
                    mode='markers+text',
                    marker=dict(color='red', size=8),
                    text=[f"{i+1}<br>{result['thk_data'][dataset_idx][i]:.1f}" for i in range(25)],
                    textposition="middle center",
                    textfont=dict(color='white', size=10),
                    name=f"Points {dataset_idx+1}",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 웨이퍼 경계선 추가
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name="Wafer Boundary",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 노즐 포인트 추가 (PHDP용)
            if result.get('show_nozzle', False):
                fig.add_trace(
                    go.Scatter(
                        x=nozzle_x,
                        y=nozzle_y,
                        mode='markers+text',
                        marker=dict(color='blue', size=10, symbol='square'),
                        text=[str(num) for num in nozzle_nums],
                        textposition="top center",
                        textfont=dict(color='yellow', size=10),
                        name="Nozzles",
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # 레이아웃 설정
        margin = (self.nozzle_radius if result.get('show_nozzle', False) else self.wafer_radius) * 0.2
        plot_radius = self.nozzle_radius if result.get('show_nozzle', False) else self.wafer_radius
        
        fig.update_layout(
            title=f"THK Data Mapping Results (Angle: {result['angle']}°)" + (" - with Nozzle Map" if result.get('show_nozzle', False) else ""),
            height=400 * rows,
            showlegend=False
        )
        
        # 축 설정
        fig.update_xaxes(
            title_text="X (mm)",
            range=[-plot_radius - margin, plot_radius + margin],
            scaleanchor="y",
            scaleratio=1
        )
        
        fig.update_yaxes(
            title_text="Y (mm)",
            range=[-plot_radius - margin, plot_radius + margin],
            scaleanchor="x",
            scaleratio=1
        )
        
        return fig

    def plot_thk_map_matplotlib(self, result, selected_datasets=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import numpy as np
        
        if selected_datasets is None:
            selected_datasets = list(range(len(result['thk_maps'])))
        
        selected_datasets = selected_datasets[:24]  # 최대 24개만
        
        # 서브플롯 개수 계산 (최대 4x6 = 24개)
        n_plots = len(selected_datasets)
        if n_plots <= 4:
            n_cols = 2
        elif n_plots <= 12:
            n_cols = 4
        else:
            n_cols = 6
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, dataset_idx in enumerate(selected_datasets):
            ax = axes[i]
            thk_map = result['thk_maps'][dataset_idx]
            
            # 히트맵 그리기
            im = ax.pcolormesh(result['xq'], result['yq'], thk_map, 
                              shading='auto', cmap=self.colormap)
            
            # 컬러바
            plt.colorbar(im, ax=ax, label='Thickness (nm)', shrink=0.8)
            
            # 측정점 표시
            ax.scatter(result['X_rot'], result['Y_rot'], 
                      c='red', s=30, marker='o', zorder=5)
            
            # 번호와 값 표시
            for j, (x, y, val) in enumerate(zip(result['X_rot'], result['Y_rot'], result['thk_data'][dataset_idx])):
                ax.text(x, y + 8, str(j+1), 
                       ha='center', va='bottom', fontsize=9, color='black', weight='bold')
                ax.text(x, y - 8, f"{val:.1f}", 
                       ha='center', va='top', fontsize=8, color='black', weight='bold')
            
            # 웨이퍼 경계선
            circle = Circle((0, 0), self.wafer_radius, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            # 노즐 포인트 표시 (PHDP용)
            if result.get('show_nozzle', False):
                nozzle_x = self.nozzle_coordinates[:, 1]
                nozzle_y = self.nozzle_coordinates[:, 2]
                nozzle_nums = self.nozzle_coordinates[:, 0].astype(int)
                
                ax.scatter(nozzle_x, nozzle_y, c='blue', s=50, marker='s', zorder=5)
                
                # 노즐 번호 표시 (방향에 따라 바깥쪽으로)
                for x, y, num in zip(nozzle_x, nozzle_y, nozzle_nums):
                    # 웨이퍼 중심에서 노즐 방향으로의 단위벡터 계산
                    length = np.sqrt(x*x + y*y)
                    if length > 0:
                        # 바깥쪽으로 20mm 더 이동
                        text_x = x + (x/length) * 20
                        text_y = y + (y/length) * 20
                    else:
                        text_x = x
                        text_y = y + 20
                    
                    ax.text(text_x, text_y, str(num), ha='center', va='center', 
                           fontsize=10, color='black', weight='bold')
            
            ax.set_title(f'THK Data {dataset_idx+1}' + (" - Nozzle Map" if result.get('show_nozzle', False) else ""), fontsize=14, weight='bold')
            ax.set_xlabel('X (mm)', fontsize=12)
            ax.set_ylabel('Y (mm)', fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 축 범위 설정 (노즐 맵인 경우 더 넓게)
            plot_radius = self.nozzle_radius if result.get('show_nozzle', False) else self.wafer_radius
            margin = plot_radius * 0.2
            ax.set_xlim(-plot_radius - margin, plot_radius + margin)
            ax.set_ylim(-plot_radius - margin, plot_radius + margin)
            
            # 통계 정보 추가
            valid_data = result['thk_data'][dataset_idx]  # 실제 25개 THK 측정값
            if len(valid_data) > 0:
                avg = np.mean(valid_data)
                rng = np.max(valid_data) - np.min(valid_data)
                std = np.std(valid_data, ddof=1)
                var = np.var(valid_data, ddof=1)
                
                stats_text = f"AVG = {avg:.1f}nm\nRNG = {rng:.1f}nm\nSTD = {std:.1f}nm\nVAR = {var:.1f}nm²"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'THK Mapping Results' + (" - with Nozzle Map" if result.get('show_nozzle', False) else ""), fontsize=16, weight='bold')
        plt.tight_layout()
        
        return fig

    def plot_wafer_map(self, result, side_names=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 20번 포인트 제외 옵션이 적용된 경우의 데이터
        exclude_point20 = result.get('exclude_point20', False)
        
        # Side1 플롯
        im1 = ax1.pcolormesh(result['xq'], result['yq'], result['zq_side1'], 
                            shading='auto', cmap=self.colormap)
        plt.colorbar(im1, ax=ax1, label='Thickness (nm)', shrink=0.8)
        
        # 측정점 표시
        ax1.scatter(result['X_rot_side1'], result['Y_rot_side1'], 
                   c='red', s=30, marker='o', zorder=5)
        
        # 번호와 값 표시
        for i, (x, y, z) in enumerate(zip(result['X_rot_side1'], result['Y_rot_side1'], result['Z'])):
            ax1.text(x, y + 8, str(int(result['data'][i, 0])), 
                    ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax1.text(x, y - 8, f"{z:.1f}", 
                    ha='center', va='top', fontsize=8, color='black', weight='bold')
        
        if side_names is None:
            side_names = {'side1': 'Side1', 'side2': 'Side2'}
        ax1.set_title(f'{side_names["side1"]} - {result["side1_angle"]}° Rotation', fontsize=14, weight='bold')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        
        # 웨이퍼 경계선
        circle1 = Circle((0, 0), self.wafer_radius, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(circle1)
        
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        margin = self.wafer_radius * 0.2
        ax1.set_xlim(-self.wafer_radius - margin, self.wafer_radius + margin)
        ax1.set_ylim(-self.wafer_radius - margin, self.wafer_radius + margin)
        
        # Side2 플롯
        im2 = ax2.pcolormesh(result['xq'], result['yq'], result['zq_side2'], 
                            shading='auto', cmap=self.colormap)
        plt.colorbar(im2, ax=ax2, label='Thickness (nm)', shrink=0.8)
        
        # 측정점 표시
        ax2.scatter(result['X_rot_side2'], result['Y_rot_side2'], 
                   c='red', s=30, marker='o', zorder=5)
        
        # 번호와 값 표시
        for i, (x, y, z1) in enumerate(zip(result['X_rot_side2'], result['Y_rot_side2'], result['Z1'])):
            ax2.text(x, y + 8, str(int(result['data'][i, 0])), 
                    ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax2.text(x, y - 8, f"{z1:.1f}", 
                    ha='center', va='top', fontsize=8, color='black', weight='bold')
        
        ax2.set_title(f'{side_names["side2"]} - {result["side2_angle"]}° Rotation', fontsize=14, weight='bold')
        ax2.set_xlabel('X (mm)', fontsize=12)
        ax2.set_ylabel('Y (mm)', fontsize=12)
        
        # 웨이퍼 경계선
        circle2 = Circle((0, 0), self.wafer_radius, fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(circle2)
        
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-self.wafer_radius - margin, self.wafer_radius + margin)
        ax2.set_ylim(-self.wafer_radius - margin, self.wafer_radius + margin)
        
        # 통계 정보 계산 및 추가 (20번 포인트 제외 옵션 적용)
        side1_data = result['Z']  # 실제 25개 측정값
        side2_data = result['Z1'] # 실제 25개 측정값
        
        # 20번 포인트 제외 시 통계 계산용 데이터 필터링
        if exclude_point20:
            # 20번 포인트 (인덱스 19) 제외
            side1_stats_data = np.concatenate([side1_data[:19], side1_data[20:]])
            side2_stats_data = np.concatenate([side2_data[:19], side2_data[20:]])
        else:
            side1_stats_data = side1_data
            side2_stats_data = side2_data
        
        # Side1 통계 정보
        if len(side1_stats_data) > 0:
            side1_avg = np.mean(side1_stats_data)
            side1_rng = np.max(side1_stats_data) - np.min(side1_stats_data)
            side1_std = np.std(side1_stats_data, ddof=1)
            side1_var = np.var(side1_stats_data, ddof=1)
            
            stats_text1 = f"AVG = {side1_avg:.1f}nm\nRNG = {side1_rng:.1f}nm\nSTD = {side1_std:.1f}nm\nVAR = {side1_var:.1f}nm²"
            
            ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Side2 통계 정보
        if len(side2_stats_data) > 0:
            side2_avg = np.mean(side2_stats_data)
            side2_rng = np.max(side2_stats_data) - np.min(side2_stats_data)
            side2_std = np.std(side2_stats_data, ddof=1)
            side2_var = np.var(side2_stats_data, ddof=1)
            
            stats_text2 = f"AVG = {side2_avg:.1f}nm\nRNG = {side2_rng:.1f}nm\nSTD = {side2_std:.1f}nm\nVAR = {side2_var:.1f}nm²"
            
            ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_wafer_map_plotly(self, result, side_names=None):
        from plotly.subplots import make_subplots
        
        # 서브플롯 생성
        if side_names is None:
            side_names = {'side1': 'Side1', 'side2': 'Side2'}
            
        exclude_point20 = result.get('exclude_point20', False)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'{side_names["side1"]} - {result["side1_angle"]}° Rotation',
                f'{side_names["side2"]} - {result["side2_angle"]}° Rotation'
            ],
            horizontal_spacing=0.1
        )
        
        # Side1 히트맵
        fig.add_trace(
            go.Heatmap(
                z=result['zq_side1'],
                x=result['xq'][0, :],
                y=result['yq'][:, 0],
                colorscale='Jet',
                showscale=True,
                colorbar=dict(title="Thickness (nm)", x=0.45),
                name="Side1"
            ),
            row=1, col=1
        )
        
        # Side1 측정점 추가
        fig.add_trace(
            go.Scatter(
                x=result['X_rot_side1'],
                y=result['Y_rot_side1'],
                mode='markers+text',
                marker=dict(color='red', size=8),
                text=[f"{int(result['data'][i, 0])}<br>{result['Z'][i]:.1f}" for i in range(25)],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name="Side1 Points",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Side2 히트맵
        fig.add_trace(
            go.Heatmap(
                z=result['zq_side2'],
                x=result['xq'][0, :],
                y=result['yq'][:, 0],
                colorscale='Jet',
                showscale=True,
                colorbar=dict(title="Thickness (nm)", x=1.02),
                name="Side2"
            ),
            row=1, col=2
        )
        
        # Side2 측정점 추가
        fig.add_trace(
            go.Scatter(
                x=result['X_rot_side2'],
                y=result['Y_rot_side2'],
                mode='markers+text',
                marker=dict(color='red', size=8),
                text=[f"{int(result['data'][i, 0])}<br>{result['Z1'][i]:.1f}" for i in range(25)],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name="Side2 Points",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 웨이퍼 경계선 추가
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.wafer_radius * np.cos(theta)
        circle_y = self.wafer_radius * np.sin(theta)
        
        # Side1 경계선
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name="Wafer Boundary",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Side2 경계선
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name="Wafer Boundary",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 레이아웃 설정
        margin = self.wafer_radius * 0.2
        fig.update_layout(
            title=f"Wafer Mapping Results ({side_names['side1']}: {result['side1_angle']}°, {side_names['side2']}: {result['side2_angle']}°)",
            height=600,
            showlegend=False
        )
        
        # 축 설정
        fig.update_xaxes(
            title_text="X (mm)",
            range=[-self.wafer_radius - margin, self.wafer_radius + margin],
            scaleanchor="y",
            scaleratio=1
        )
        
        fig.update_yaxes(
            title_text="Y (mm)",
            range=[-self.wafer_radius - margin, self.wafer_radius + margin],
            scaleanchor="x",
            scaleratio=1
        )
        
        return fig

class HistoryManager:
    def __init__(self):
        if 'history_data' not in st.session_state:
            st.session_state.history_data = []
    
    def add_entry(self, equipment_type, unit, lp, side1_angle, side2_angle, 
                  side1_data, side2_data, thk_data=None, result=None):
        entry = {
            'timestamp': datetime.now(),
            'equipment_type': equipment_type,
            'unit': unit,
            'lp': lp,
            'side1_angle': side1_angle,
            'side2_angle': side2_angle,
            'side1_data': side1_data.tolist() if side1_data is not None else None,
            'side2_data': side2_data.tolist() if side2_data is not None else None,
            'thk_data': [data.tolist() for data in thk_data] if thk_data else None,
            'result_type': 'wafer_map' if result and 'side1_angle' in result else 'thk_map'
        }
        
        st.session_state.history_data.append(entry)
        self.cleanup_old_entries()
    
    def cleanup_old_entries(self):
        cutoff_time = datetime.now() - timedelta(hours=1)
        st.session_state.history_data = [
            entry for entry in st.session_state.history_data 
            if entry['timestamp'] > cutoff_time
        ]
    
    def get_recent_entries(self):
        self.cleanup_old_entries()
        return sorted(st.session_state.history_data, 
                     key=lambda x: x['timestamp'], reverse=True)
    
    def load_entry(self, index):
        entries = self.get_recent_entries()
        if 0 <= index < len(entries):
            return entries[index]
        return None

def get_image_download_link(fig, filename):
    """matplotlib 그림을 다운로드 링크로 변환"""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    b64 = base64.b64encode(img.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">이미지 다운로드</a>'
    return href

def main():
    st.set_page_config(
        page_title="웨이퍼 매핑 도구",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("웨이퍼 매핑 도구")
    st.markdown("---")
    
    # 초기화
    eq_config = EquipmentConfig()
    wm = WaferMapping()
    history = HistoryManager()
    
    # 사이드바 설정
    st.sidebar.header("설비 설정")
    
    # 설비 선택
    equipment_type = st.sidebar.selectbox(
        "설비 선택",
        ["PHDP", "SIOC", "TEOS", "MIR3000", "TES"],
        help="분석할 설비 타입을 선택하세요"
    )
    
    # 호기 선택
    eq_info = eq_config.equipment_types[equipment_type]
    unit = st.sidebar.selectbox(
        "호기 선택",
        eq_info['units'],
        help=f"{equipment_type} 설비의 호기를 선택하세요"
    )
    
    # PHDP용 노즐 맵 옵션
    show_nozzle_map = False
    if equipment_type == 'PHDP' and eq_info.get('supports_nozzle', False):
        show_nozzle_map = st.sidebar.checkbox(
            "N/Z Tune Map 표시",
            help="32개 노즐 포인트를 함께 표시합니다"
        )
    
    # TEOS용 20번 포인트 제외 옵션
    exclude_point20 = False
    if equipment_type == 'TEOS' and eq_info.get('supports_exclude_point20', False):
        exclude_point20 = st.sidebar.checkbox(
            "20번 포인트 제외 (Range 계산)",
            help="통계 계산에서 20번 포인트를 제외합니다"
        )
    
    # 이력 섹션
    st.sidebar.markdown("---")
    st.sidebar.subheader("이력")
    recent_entries = history.get_recent_entries()
    
    if recent_entries:
        history_options = [f"{entry['timestamp'].strftime('%H:%M')} - {entry['equipment_type']} {entry['unit']}" 
                          for entry in recent_entries]
        selected_history = st.sidebar.selectbox(
            "최근 이력 (1시간 내)",
            ["새로운 분석"] + history_options,
            help="최근 1시간 내 분석 이력을 선택하세요"
        )
        
        if selected_history != "새로운 분석":
            history_idx = history_options.index(selected_history)
            if st.sidebar.button("선택한 이력 불러오기"):
                loaded_entry = history.load_entry(history_idx)
                if loaded_entry:
                    # 세션 상태에 저장하여 UI 업데이트
                    for key, value in loaded_entry.items():
                        if key != 'timestamp':
                            st.session_state[f'loaded_{key}'] = value
                    st.rerun()
    
    # LP 설정 (SIOC, HITEOS만)
    lp_choice = None
    if eq_info['has_sides']:
        lp_choice = st.sidebar.selectbox(
            "Load Port 선택",
            ["LP1", "LP2", "LP3"],
            help="Load Port를 선택하세요"
        )
        
        # 신규 EFEM 여부 표시
        if eq_config.is_new_efem(unit):
            st.sidebar.info(f"{unit}은 신규 EFEM 설비입니다.")
    
    # 각도 설정
    st.sidebar.subheader("각도 설정")
    
    side1_angle = 0
    side2_angle = 0
    
    if eq_info.get('angle_fixed', False):
        side1_angle = eq_info.get('fixed_angle', 0)
        side2_angle = eq_info.get('fixed_angle', 0)
        st.sidebar.info(f"각도 고정: {eq_info.get('fixed_angle', 0)}°")
    
    elif eq_info['has_sides'] and lp_choice:
        # 기본 각도 가져오기
        default_angles = eq_config.get_default_angles(equipment_type, unit, lp_choice)
        
        st.sidebar.markdown(f"**기본 각도 (LP{lp_choice[-1]})**")
        st.sidebar.text(f"Side1: {default_angles['side1']}°")
        st.sidebar.text(f"Side2: {default_angles['side2']}°")
        
        # 각도 조정 옵션
        angle_mode = st.sidebar.radio(
            "각도 입력 방식",
            ["기본값 사용", "임의 조정"],
            help="기본값을 사용하거나 직접 조정하세요"
        )
        
        if angle_mode == "기본값 사용":
            side1_angle = default_angles['side1']
            side2_angle = default_angles['side2']
        else:
            side1_angle = st.sidebar.number_input(
                "Side1 각도 (도)",
                min_value=-360.0,
                max_value=360.0,
                value=float(default_angles['side1']),
                step=0.1,
                help="Side1 회전 각도를 입력하세요"
            )
            
            side2_angle = st.sidebar.number_input(
                "Side2 각도 (도)",
                min_value=-360.0,
                max_value=360.0,
                value=float(default_angles['side2']),
                step=0.1,
                help="Side2 회전 각도를 입력하세요"
            )
    
    # 메인 영역
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("데이터 입력")
        
        # THK 데이터 처리 (PHDP, MIR3000, TEAP360 특별 처리)
        if eq_info.get('supports_thk', False) or eq_config.is_teap360_special(unit):
            st.subheader("THK 데이터 입력")
            
            # THK 데이터셋 개수 선택
            default_thk_count = 2
            num_thk = st.number_input(
                "THK 데이터셋 개수",
                min_value=1,
                max_value=eq_info.get('max_thk_count', 24),
                value=default_thk_count,
                help=f"분석할 THK 데이터셋 개수 (최대 {eq_info.get('max_thk_count', 24)}개)"
            )
            
            # 데이터 입력 방식
            input_method = st.radio(
                "데이터 입력 방식",
                ["데모 데이터", "직접 입력"],
                help="데모 데이터 또는 직접 입력을 선택하세요"
            )
            
            thk_datasets = []
            
            if input_method == "데모 데이터":
                st.info(f"데모 데이터를 사용합니다. ({num_thk}개 데이터셋)")
                np.random.seed(42)
                for i in range(num_thk):
                    thk_data = np.random.normal(2500 + i*50, 30, 25)
                    thk_datasets.append(thk_data)
            else:
                all_data_valid = True
                
                # 페이징 설정 (한 페이지에 2개씩)
                items_per_page = 2
                total_pages = (num_thk + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    page = st.selectbox(
                        f"페이지 선택 (총 {total_pages}페이지, 페이지당 {items_per_page}개)",
                        range(1, total_pages + 1),
                        format_func=lambda x: f"페이지 {x} (THK Data {(x-1)*items_per_page+1}-{min(x*items_per_page, num_thk)})"
                    )
                else:
                    page = 1
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, num_thk)
                
                # 현재 페이지의 데이터 입력
                for i in range(start_idx, end_idx):
                    st.subheader(f"THK Data {i+1}")
                    thk_text = st.text_area(
                        f"THK Data {i+1} 입력 (25개, 각 줄에 하나씩)",
                        height=150,
                        placeholder="2500.1\n2501.2\n2502.3\n...",
                        key=f"thk_{i}",
                        help="숫자 데이터를 한 줄에 하나씩 입력하세요"
                    )
                
                # 전체 데이터 검증 (세션 상태 사용)
                if f'thk_datasets_{equipment_type}_{unit}' not in st.session_state:
                    st.session_state[f'thk_datasets_{equipment_type}_{unit}'] = {}
                
                current_datasets = st.session_state[f'thk_datasets_{equipment_type}_{unit}']
                
                # 현재 페이지 데이터 저장
                for i in range(start_idx, end_idx):
                    thk_text = st.session_state.get(f"thk_{i}", "")
                    if thk_text:
                        try:
                            thk_values = [float(x.strip()) for x in thk_text.strip().split('\n') if x.strip()]
                            if len(thk_values) == 25:
                                current_datasets[i] = np.array(thk_values)
                            else:
                                if i in current_datasets:
                                    del current_datasets[i]
                                st.error(f"THK Data {i+1}: 정확히 25개의 값이 필요합니다. (현재: {len(thk_values)})")
                                all_data_valid = False
                        except ValueError:
                            if i in current_datasets:
                                del current_datasets[i]
                            st.error(f"THK Data {i+1}: 숫자 형식이 올바르지 않습니다.")
                            all_data_valid = False
                    else:
                        if i in current_datasets:
                            del current_datasets[i]
                        all_data_valid = False
                
                # 전체 데이터셋 상태 확인
                thk_datasets = []
                for i in range(num_thk):
                    if i in current_datasets:
                        thk_datasets.append(current_datasets[i])
                    else:
                        all_data_valid = False
                
                # 데이터 입력 상태 표시
                if total_pages > 1:
                    completed_count = len(current_datasets)
                    st.info(f"전체 진행상황: {completed_count}/{num_thk}개 데이터셋 입력 완료")
                    
                    if completed_count < num_thk:
                        remaining = []
                        for i in range(num_thk):
                            if i not in current_datasets:
                                remaining.append(f"THK Data {i+1}")
                        st.warning(f"입력 필요: {', '.join(remaining)}")
                
                if len(thk_datasets) == num_thk and all_data_valid:
                    st.success("모든 THK 데이터 입력 완료!")
        
        # Side1/Side2 데이터 처리 (SIOC, TEOS)
        elif not eq_info.get('supports_thk', False) and not eq_config.is_teap360_special(unit):
            side1_data = None
            side2_data = None
            
            input_method = st.radio(
                "데이터 입력 방식",
                ["데모 데이터", "직접 입력"],
                help="데모 데이터 또는 직접 입력을 선택하세요"
            )
            
            if input_method == "데모 데이터":
                st.info("데모 데이터를 사용합니다.")
                np.random.seed(42)
                side1_data = np.random.normal(2500, 50, 25)
                side2_data = np.random.normal(2600, 50, 25)
                
            else:
                side_names = eq_info.get('side_names', {'side1': 'Side1', 'side2': 'Side2'})
                st.subheader(f"{side_names['side1']} 두께 데이터")
                side1_text = st.text_area(
                    f"{side_names['side1']} 값 입력 (25개, 각 줄에 하나씩)",
                    height=150,
                    placeholder="2500.1\n2501.2\n2502.3\n...",
                    help="숫자 데이터를 한 줄에 하나씩 입력하세요"
                )
                
                st.subheader(f"{side_names['side2']} 두께 데이터")
                side2_text = st.text_area(
                    f"{side_names['side2']} 값 입력 (25개, 각 줄에 하나씩)",
                    height=150,
                    placeholder="2600.1\n2601.2\n2602.3\n...",
                    help="숫자 데이터를 한 줄에 하나씩 입력하세요"
                )
            
                if side1_text and side2_text:
                    try:
                        side1_values = [float(x.strip()) for x in side1_text.strip().split('\n') if x.strip()]
                        side2_values = [float(x.strip()) for x in side2_text.strip().split('\n') if x.strip()]
                        
                        if len(side1_values) == 25 and len(side2_values) == 25:
                            side1_data = np.array(side1_values)
                            side2_data = np.array(side2_values)
                            st.success("데이터 입력 완료!")
                        else:
                            st.error(f"각 면마다 정확히 25개의 값이 필요합니다. (Side1: {len(side1_values)}, Side2: {len(side2_values)})")
                    except ValueError:
                        st.error("숫자 형식이 올바르지 않습니다.")
    
    with col2:
        st.header("웨이퍼 맵")
        
        # THK 데이터 시각화
        if (eq_info.get('supports_thk', False) or eq_config.is_teap360_special(unit)) and 'thk_datasets' in locals() and len(thk_datasets) > 0:
            result = wm.create_thk_map(thk_datasets, angle=side1_angle, show_nozzle=show_nozzle_map)
            
            # 표시할 데이터셋 선택
            if len(thk_datasets) > 1:
                st.subheader("표시할 데이터셋 선택")
                dataset_options = [f"THK Data {i+1}" for i in range(len(thk_datasets))]
                
                # 전체 선택 버튼
                if st.button("전체 선택"):
                    st.session_state.selected_all = True
                if st.button("전체 해제"):
                    st.session_state.selected_all = False
                
                default_selection = dataset_options if st.session_state.get('selected_all', False) else dataset_options[:min(2, len(dataset_options))]
                
                selected_datasets_names = st.multiselect(
                    f"데이터셋을 선택하세요 (최대 {min(24, len(dataset_options))}개)",
                    dataset_options,
                    default=default_selection,
                    help="동시에 표시할 데이터셋을 선택하세요"
                )
                
                selected_datasets = [dataset_options.index(name) for name in selected_datasets_names]
            else:
                selected_datasets = [0]
            
            if selected_datasets:
                # 시각화 옵션
                viz_option = st.radio(
                    "시각화 옵션",
                    ["정적 그래프 (Matplotlib)", "인터랙티브 그래프 (Plotly)"],
                    horizontal=True
                )
                
                # THK 매핑 결과 표시
                if viz_option == "정적 그래프 (Matplotlib)":
                    fig_thk = wm.plot_thk_map_matplotlib(result, selected_datasets)
                    if fig_thk:
                        st.pyplot(fig_thk)
                else:
                    fig_thk = wm.plot_thk_map_plotly(result, selected_datasets)
                    if fig_thk:
                        st.plotly_chart(fig_thk, use_container_width=True)
                
                # 통계 정보
                st.subheader("통계 정보")
                for idx in selected_datasets:
                    col_stat = st.columns(1)[0]
                    with col_stat:
                        st.markdown(f"**THK Data {idx+1}**")
                        thk_map = result['thk_maps'][idx]
                        st.metric(f"평균", f"{np.nanmean(thk_map):.2f} nm")
                        st.metric(f"표준편차", f"{np.nanstd(thk_map):.2f} nm")
                        st.metric(f"범위", f"{np.nanmin(thk_map):.1f} ~ {np.nanmax(thk_map):.1f} nm")
                
                # 이력 저장
                history.add_entry(
                    equipment_type, unit, lp_choice, side1_angle, side2_angle,
                    None, None, thk_datasets, result
                )
            else:
                st.info("표시할 데이터셋을 선택하세요.")
        
        # Side1/Side2 데이터 시각화
        elif not eq_info.get('supports_thk', False) and 'side1_data' in locals() and side1_data is not None and side2_data is not None:
            result = wm.create_wafer_map(side1_angle, side2_angle, side1_data, side2_data, exclude_point20)
            
            # 시각화 옵션
            viz_option = st.radio(
                "시각화 옵션",
                ["정적 그래프 (Matplotlib)", "인터랙티브 그래프 (Plotly)"],
                horizontal=True
            )
            
            side_names = eq_info.get('side_names', {'side1': 'Side1', 'side2': 'Side2'})
            if viz_option == "정적 그래프 (Matplotlib)":
                fig = wm.plot_wafer_map(result, side_names)
                st.pyplot(fig)
                
                filename = f"wafer_map_{equipment_type}_{unit}_S1_{result['side1_angle']}_S2_{result['side2_angle']}.png"
                if exclude_point20:
                    filename = filename.replace('.png', '_ex20.png')
                download_link = get_image_download_link(fig, filename)
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                fig_plotly = wm.plot_wafer_map_plotly(result, side_names)
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            # 통계 정보 (20번 포인트 제외 옵션 적용)
            st.subheader("통계 정보")
                
            side1_stats_data = result['Z']
            side2_stats_data = result['Z1']
            
            # 20번 포인트 제외 시 통계용 데이터 필터링
            if exclude_point20:
                side1_stats_data = np.concatenate([side1_stats_data[:19], side1_stats_data[20:]])
                side2_stats_data = np.concatenate([side2_stats_data[:19], side2_stats_data[20:]])
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown(f"**{side_names['side1']}**")
                st.metric("평균", f"{np.mean(side1_stats_data):.2f} nm")
                st.metric("표준편차", f"{np.std(side1_stats_data, ddof=1):.2f} nm")
                st.metric("범위", f"{np.min(side1_stats_data):.1f} ~ {np.max(side1_stats_data):.1f} nm")
            
            with col_stat2:
                st.markdown(f"**{side_names['side2']}**")
                st.metric("평균", f"{np.mean(side2_stats_data):.2f} nm")
                st.metric("표준편차", f"{np.std(side2_stats_data, ddof=1):.2f} nm")
                st.metric("범위", f"{np.min(side2_stats_data):.1f} ~ {np.max(side2_stats_data):.1f} nm")
            
            # 이력 저장
            history.add_entry(
                equipment_type, unit, lp_choice, side1_angle, side2_angle,
                side1_data, side2_data, None, result
            )
            
        else:
            if eq_info.get('supports_thk', False) or eq_config.is_teap360_special(unit):
                st.info("THK 데이터를 입력하면 웨이퍼 맵이 표시됩니다.")
            else:
                st.info("Side1/Side2 데이터를 입력하면 웨이퍼 맵이 표시됩니다.")
    
    # 푸터
    st.markdown("---")
    additional_info = ""
    if equipment_type == 'PHDP' and show_nozzle_map:
        additional_info += " | N/Z Tune Map"
    if equipment_type == 'TEOS' and exclude_point20:
        additional_info += " | 20pt 제외"
    
    st.markdown(f"**웨이퍼 매핑 도구** | {equipment_type} {unit} 분석{additional_info}")

if __name__ == "__main__":
    main()
