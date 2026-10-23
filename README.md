# rustfox

Firefox-подход к оптимизации браузерных hot-path на Rust: цикл
**бенчмарк → профилирование → оптимизация → бенчмарк**, крутящийся локально и в CI.

## Что внутри

| крейт | назначение |
|---|---|
| `rustfox-dom` | arena-DOM с интернированием строк (теги/классы → `u32`) — сравнения в hot path целочисленные, как atom table в Stylo |
| `rustfox-style` | парсер CSS-селекторов + матчеры: `naive` (база), `optimized` (bloom-фильтр предков), `indexed` (rule bucketing), `fast` (v0.8: бит-параллельный NFA + AVX2 + **инкрементальный restyle** по dirty-поддеревьям на preorder), `fast_okc` (direct-mapped ok-cache) |
| `rustfox-simd` | AVX2/SSE2/SWAR байт-сканирование с кэшем детекции фич и гибридным диспатчем по длине — memchr-слой всех остальных ядер |
| `rustfox-url` | zero-copy URL-парсер (kernel nsStandardURL): **однопроходный bitmap-движок** (7 структурных байтов за один SIMD-проход), только Range-индексы, ноль аллокаций |
| `rustfox-html` | zero-copy HTML-токенизатор (kernel nsHtml5Tokenizer): pull-API + fused-движок (плоская арена событий) + **streaming `for_each_event`** (колбэк в момент скана, без арены и второго прохода), один сканер на три API, 0 аллокаций |
| `rustfox-http1` | zero-copy HTTP/1.1 заголовки (kernel necko): two-pass с записью концов строк за один SIMD-скан, статус сдвигами, inline×32 + spill |

Текущие результаты (локальный прогон, полный лог — [WORKLOG.md](WORKLOG.md),
свежие цифры — [bench/latest.json](bench/latest.json)):

| ядро | rf | naive/база | выигрыш |
|---|---|---|---|
| style_pass @2k | 287 µs | 2.31 ms | **8.04x** |
| style_pass @10k | 1.60 ms | 11.8 ms | **7.35x** |
| restyle 1% dirty @10k | 218 µs | 1.59 ms (full recalc) | **7.3x** |
| restyle 0.1% dirty @10k | 168 µs | 1.59 ms | **9.4x** |
| url/parse (202 KiB, bitmap) | 102 µs | 172 µs | **1.68x** |
| http1/parse (1 MiB) | 671 µs | 1.38 ms | **2.05x** |
| html/tokenize prose (callback) | 88 µs | 178 µs | **2.03x** |
| html/tokenize dense (callback) | 384 µs | 282 µs | 0.73x (честно: скан ≈ naive на ранах 1–15B; лучший rustfox-время: было 410 µs у fused) |
| simd find_byte 4k | 83 ns | 1.17 µs | **14.0x** |

Архитектура `fast` — топ-даун бит-параллельный автомат: переход O(W) u128/AVX2
операций на узел вместо обходов предков, SoA-снапшот DOM переиспользуется между
проходами, parent-стейт читается из store без копий. Датафлоу строго
parent → child, поэтому v0.8 умеет инкрементальный restyle: класс изменился —
пересчитывается только поддерево (поддерево = непрерывный диапазон в
preorder), чистые узлы копируются memcpy, парити с полным проходом —
бит-в-байт (matched sets и StateStore).

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
