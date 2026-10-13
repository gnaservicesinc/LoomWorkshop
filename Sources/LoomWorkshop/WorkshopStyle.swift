import SwiftUI

enum WorkshopStyle {
    static let accent = Color(red: 0.12, green: 0.64, blue: 0.52)
    static let radius: CGFloat = 16
}

struct WorkshopCard<Content: View>: View {
    let number: Int
    let title: String
    let subtitle: String
    private let content: Content
    init(_ number: Int, _ title: String, subtitle: String, @ViewBuilder content: () -> Content) {
        self.number = number; self.title = title; self.subtitle = subtitle; self.content = content()
    }
    var body: some View {
        HStack(alignment: .top, spacing: 20) {
            Text(number, format: .number).font(.headline).foregroundStyle(WorkshopStyle.accent)
                .frame(width: 34, height: 34).background(WorkshopStyle.accent.opacity(0.1), in: Circle())
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.headline)
                    Text(subtitle).font(.callout).foregroundStyle(.secondary)
                }
                content
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .background(.background, in: RoundedRectangle(cornerRadius: WorkshopStyle.radius))
        .overlay(RoundedRectangle(cornerRadius: WorkshopStyle.radius).strokeBorder(Color(nsColor: .separatorColor).opacity(0.55)))
    }
}

struct WorkshopMark: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 15).fill(WorkshopStyle.accent.gradient)
            VStack(spacing: 5) {
                ForEach(0..<3) { row in
                    HStack(spacing: 5) {
                        ForEach(0..<3) { column in
                            RoundedRectangle(cornerRadius: 2).fill(.white.opacity(row == column ? 1 : 0.45))
                                .frame(width: 8, height: 8)
                        }
                    }
                }
            }
        }.accessibilityHidden(true)
    }
}

struct WorkshopPath: View {
    let path: URL
    let symbol: String
    var body: some View {
        Label(path.path, systemImage: symbol).font(.callout).foregroundStyle(.secondary)
            .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
    }
}
