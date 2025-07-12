
class DashboardCharts {
    constructor() {
        this.chartData = null;
        this.speedChart = null;
        this.directionChart = null;
        this.loadChartData();
    }

    async loadChartData() {
        try {
            console.log("Loading chart data...");
            const response = await fetch('/api/charts');
            const data = await response.json();
            this.chartData = data;
            console.log("Chart data received:", data);
            this.renderCharts();
        } catch (error) {
            console.error("Failed to fetch chart data:", error);
            this.showError('speed');
            this.showError('direction');
        }
    }

    renderCharts() {
        if (!this.chartData) return;
        this.renderSpeedChart();
        this.renderDirectionChart();
    }

    renderSpeedChart() {
        const data = this.chartData.speed_histogram;
        const loadingEl = document.getElementById('speedChartLoading');
        const errorEl = document.getElementById('speedChartError');
        const canvas = document.getElementById('speedChart');

        if (loadingEl) loadingEl.style.display = 'none';

        if (!canvas || !data || !data.labels || !data.data || data.data.every(d => d === 0)) {
            if (errorEl) errorEl.style.display = 'block';
            return;
        }

        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        this.speedChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Detections',
                    data: data.data,
                    backgroundColor: '#3e95cd'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    renderDirectionChart() {
        const data = this.chartData.direction_breakdown;
        const loadingEl = document.getElementById('directionChartLoading');
        const errorEl = document.getElementById('directionChartError');
        const canvas = document.getElementById('directionChart');

        if (loadingEl) loadingEl.style.display = 'none';

        if (!canvas || !data || !data.labels || !data.data || data.data.every(d => d === 0)) {
            if (errorEl) errorEl.style.display = 'block';
            return;
        }

        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        this.directionChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Directions',
                    data: data.data,
                    backgroundColor: ['#007bff', '#ffc107', '#28a745', '#6c757d']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    showError(type) {
        const loadingEl = document.getElementById(`${type}ChartLoading`);
        const errorEl = document.getElementById(`${type}ChartError`);
        const canvas = document.getElementById(`${type}Chart`);
        if (loadingEl) loadingEl.style.display = 'none';
        if (canvas) canvas.style.display = 'none';
        if (errorEl) errorEl.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.dashboardCharts = new DashboardCharts();
});
