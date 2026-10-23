# rustfox

Firefox-подход к оптимизации браузерных hot-path на Rust: цикл
**бенчмарк → профилирование → оптимизация → бенчмарк**, крутящийся локально и в CI.

## Что внутри

| крейт | назначение |
|---|---|
| `rustfox-dom` | arena-DOM с интернированием строк (теги/классы → `u32`) — сравнения в hot path целочисленные, как atom table в Stylo |
| `rustfox-style` | парсер CSS-селекторов + матчеры: `naive` (база), `optimized` (bloom-фильтр предков), `indexed` (rule bucketing), `fast` (v0.7: бит-параллельный NFA + AVX2 + zero-copy parent reads), `fast_okc` (direct-mapped ok-cache) |
| `rustfox-simd` | AVX2/SSE2/SWAR байт-сканирование с кэшем детекции фич и гибридным диспатчем по длине — memchr-слой всех остальных ядер |
| `rustfox-url` | zero-copy URL-парсер (kernel nsStandardURL): только Range-индексы, ноль аллокаций |
| `rustfox-html` | zero-copy SIMD HTML-токенизатор (kernel nsHtml5Tokenizer): scratch-Attrs, 0 аллокаций на токен |
| `rustfox-http1` | zero-copy HTTP/1.1 заголовки (kernel necko): статус сдвигами, inline×32 + spill |

Текущие результаты (локальный прогон, полный лог — [WORKLOG.md](WORKLOG.md),
свежие цифры — [bench/latest.json](bench/latest.json)):

| ядро | rf | naive | выигрыш |
|---|---|---|---|
| style_pass @2k | 289 µs | 2.30 ms | **7.96x** |
| style_pass @10k | 1.60 ms | 12.0 ms | **7.52x** |
| http1/parse (1 MiB) | 753 µs | 1.39 ms | **1.85x** |
| html/tokenize prose (489 KiB) | 100 µs | 186 µs | **1.86x** |
| html/tokenize dense | 441 µs | 321 µs | 0.73x (честно: token-dense разметка ниже SIMD break-even) |
| url/parse (202 KiB) | 145 µs | 166 µs | **1.15x** |
| simd find_byte 4k | 84 ns | 1.18 µs | **14.1x** |

Архитектура `fast` — топ-даун бит-параллельный автомат: переход O(W) u128/AVX2
операций на узел вместо обходов предков, SoA-снапшот DOM переиспользуется между
проходами, parent-стейт читается из store без копий.

## Быстрый старт

```bash
cargo test          # корректность: naive == optimized == indexed на фикстурах
cargo bench         # criterion: naive vs bloom vs indexed
```

## CI

| workflow | что делает | расписание |
|---|---|---|
| `ci.yml` | fmt, clippy `-D warnings`, тесты, компиляция бенчей | на каждый push/PR |
| `bench.yml` | criterion-прогон, результаты коммитятся в ветку `bench-results` (трекинг дрейфа производительности) | cron `0 3 * * *` + вручную |
| `profile.yml` | perf + flamegraph SVG артефактом — искать следующее горячее место | cron `0 4 * * 1` + вручную |

Смежный репозиторий: [x7717/gecko-dev](https://github.com/x7717/gecko-dev) — форк
официального зеркала Firefox с CI для Rust-компонентов (WebRender) и планом
оптимизаций в [FIREFOX_RUST_PLAN.md](https://github.com/x7717/gecko-dev/blob/master/FIREFOX_RUST_PLAN.md).

## Честные рамки проекта

Полный «форк Firefox на Rust» — это проект масштаба Servo (годы работы большой
команды); Gecko уже содержит большие Rust-компоненты (Stylo, WebRender, neqo).
Этот репозиторий делает то, что реально и измеримо: воспроизводит техники
production-движков (rule hash, ancestor bloom filter, интернинг, flat-структуры)
на настоящем Rust-коде, гоняет их через автоматизированный цикл оптимизаций
с числами до/после на каждом шаге и переносит проверенные техники в апстрим-план.
