// =====================================================
//  DEVICE ENCLOSURE v3 — ESP8266 + Sensors
//  Units: mm
// =====================================================

$fn = 60;
W = 2.0;

BX = 80; BY = 90; BZ = 20;

SH = 7; SR = 2.5;
S_RX = BX - W - SR;
S_LX = S_RX - 60;
S_FY = W + SR;
S_BY = BY - W - SR;

D_X  = 5;  D_Y  = 5;
D_W  = 32; D_H  = 62;
SHELF_Z = 11;
SHELF_D = 10;

// ──── Виступний рельєф ────
ENG_D = 0.6;    // висота виступу, мм

// Ємнісна кнопка підсвічування — ліворуч від ІК
BTN_X = 9;
BTN_Z = 10;
BTN_R = 5;

// ─────────────────────────────────────────────────
//  Іконка лампочки (2D) — для виступного рельєфу
// ─────────────────────────────────────────────────
module lightbulb_2d() {
    union() {
        // Кільце-рамка (тільки кільце без внутрішньої порожнини для надійності)
        circle(r = BTN_R - 0.2);
        
        // Скляна куля
        translate([0, 0.5])
            circle(r = 1.9);

        // Конусний перехід скло → цоколь
        polygon(points=[
            [-1.9, 0.8],
            [1.9, 0.8],
            [0.75, -0.75],
            [-0.75, -0.75]
        ]);

        // Цоколь (дві сходинки)
        translate([-1.05, -1.85])
            square([2.1, 0.9]); 
        translate([-0.75, -2.55])
            square([1.5, 0.8]);

        // Нитка розжарювання (хрест)
        translate([-0.15, -0.5])
            square([0.3, 1.2]);
        translate([-0.7, -0.1])
            square([1.4, 0.3]);
    }
}

// =====================================================
//  1. КОРПУС
// =====================================================
module body() {
    union() {
        difference() {
            cube([BX, BY, BZ]);

            // Внутрішня порожнина
            translate([W, W, W])
                cube([BX - 2*W, BY - 2*W, BZ - W + 0.1]);

            // ІК-приймач
            translate([21, -0.1, 7])
                rotate([-90, 0, 0])
                cylinder(h = W + 0.2, r = 3);

            // Micro USB
            translate([BX - 12 - 4.5, -0.1, 14 - 2])
                cube([9, W + 0.2, 4]);
        }

        // ── Виступна іконка лампочки на передній стінці ──
        // rotate([-90, 180, 0]) → local +Z вказує в -Y (назовні від корпусу),
        // тому translate до y=0 і extrude виступає назовні
        translate([BTN_X, 0, BTN_Z])
            rotate([-90, 180, 0])
            linear_extrude(height = ENG_D)
            lightbulb_2d();

        // ── Дві стойки, що стоять на дні ──
        translate([W, D_Y, W])
            for (py = [SR + 1, D_H - SR - 1])
                translate([SR + 1, py, 0])
                    difference() {
                        cylinder(h = 15, r = SR);
                        translate([0, 0, -0.01])
                            cylinder(h = 15 + 0.02, r = 1.65);
                    }

        // Стойки з різьбовими отворами M4
        for (px = [S_LX, S_RX])
            for (py = [S_FY, S_BY])
                translate([px, py, W])
                    difference() {
                        cylinder(h = SH, r = SR);
                        cylinder(h = SH + 0.01, r = 1.65);
                    }
    }
}

// =====================================================
//  2. ВЕРХНЯ КРИШКА
// =====================================================
module lid() {
    union() {
        difference() {
            cube([BX, BY, W]);

            // Вікно дисплея
            translate([10, 5, -0.1])
                cube([D_W, D_H, W + 0.2]);

            // Датчик
            translate([50, 78, -0.1])
                cylinder(h = W + 0.2, r = 2);

            // Датчик газу MQ
            translate([25, 78, -0.1])
                cylinder(h = W + 0.2, r = 10);

            // DHT (температура/вологість)
            translate([BX - 2 - 13, BY - 3 - 16, -0.1])
                cube([13, 16, W + 0.2]);
        }

        // ── Виступний текст — права колонка кришки ──
        // z = W → виступ іде нагору над поверхнею кришки

        // Рядок 1
        translate([64, 60, W])
            rotate([0, 0, -90])
            linear_extrude(height = ENG_D)
            text("Моніторінг SSN.pp.ua",
                 size   = 3.4,
                 font   = "Liberation Sans:style=Regular",
                 halign = "left");

        // Рядок 2
        translate([58, 60, W])
            rotate([0, 0, -90])
            linear_extrude(height = ENG_D)
            text("ДУІТЗ — Артем Іщенко",
                 size   = 3.4,
                 font   = "Liberation Sans:style=Regular",
                 halign = "left");
    }
}

// =====================================================
//  РЕНДЕР
// =====================================================
color("Silver",    0.85) body();
color("LightBlue", 0.85) translate([BX + 15, 0, 0]) lid();
